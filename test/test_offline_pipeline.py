"""
SLT Offline Integration Test (Stage 1 -> Stage 2 -> Stage 3)
Validates the end-to-end translation pipeline using pre-extracted .npy files.
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Add src/ to path so we can import the Stage 2 model class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_stage_2 import SLTStage2CTC 

# Configuration Paths
STAGE2_WEIGHTS = "weights/stage2_best_model.pth"
STAGE3_DIR = "weights/slt_final_t5_model"
ASL_DATA_DIR = "ASL_landmarks_float16" 

# Use MPS for Apple Silicon M4, fallback to CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BLANK_IDX = 0 # Standard CTC Blank index


def _normalize_for_compare(s: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace for fair comparison."""
    s = s.lower().strip()
    for p in ".,?!;:'\"":
        s = s.replace(p, " ")
    return " ".join(s.split())


# =========================================================
# 1. The Data "Input" (Mocking the User)
# =========================================================
def resolve_gloss_to_file(gloss: str, data_dir: str) -> str:
    """Find one .npy file whose name starts with GLOSS_ (e.g. TOMORROW_xxx.npy)."""
    prefix = f"{gloss}_"
    for f in os.listdir(data_dir):
        if f.endswith(".npy") and f.startswith(prefix):
            return f
    raise FileNotFoundError(f"No file found for gloss '{gloss}' in {data_dir} (expected prefix: {prefix}*.npy)")


def load_and_concatenate_signs(sign_filenames: list, data_dir: str) -> tuple:
    """
    Loads individual 32-frame .npy files and concatenates them into a continuous sequence.
    Returns:
        batch_tensor: Shape [1, N*32, 42, 10]
        x_lens: Shape [1] containing the total frame count.
    """
    arrays = []
    print(f"📂 Loading {len(sign_filenames)} signs from {data_dir}...")
    
    for filename in sign_filenames:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing file: {filepath}")
            
        # Load [32, 42, 10] array
        arr = np.load(filepath).astype(np.float32) 
        if arr.shape != (32, 42, 10):
            print(f"⚠️ Warning: {filename} has shape {arr.shape}, expected (32, 42, 10)")
        arrays.append(arr)
        
    # Concatenate along the time dimension (axis 0) -> [N*32, 42, 10]
    continuous_sequence = np.concatenate(arrays, axis=0)
    
    # Add batch dimension -> [1, N*32, 42, 10]
    batch_tensor = torch.from_numpy(continuous_sequence).unsqueeze(0).to(DEVICE)
    
    # Length of the sequence (N*32)
    x_lens = torch.tensor([continuous_sequence.shape[0]], dtype=torch.long).to(DEVICE)
    
    print(f"✅ Data Concatenated! Sequence Shape: {batch_tensor.shape}")
    return batch_tensor, x_lens

# =========================================================
# 2. The Stage 1 + 2 "Transcriber"
# =========================================================
def load_transcriber():
    """Loads the frozen DS-GCN + BiLSTM CTC model."""
    print("⏳ Loading Stage 2 Transcriber...")
    if not os.path.exists(STAGE2_WEIGHTS):
         raise FileNotFoundError(f"Cannot find checkpoint: {STAGE2_WEIGHTS}")
         
    ckpt = torch.load(STAGE2_WEIGHTS, map_location=DEVICE, weights_only=False)
    
    # Safely extract vocab dictionaries (handle JSON integer-to-string key issues)
    idx_to_gloss = {int(k): v for k, v in ckpt.get('idx_to_gloss', {}).items()}
    vocab_size = ckpt.get('vocab_size', len(idx_to_gloss))
    
    # Instantiate exact architecture from train_stage_2.py
    model = SLTStage2CTC(vocab_size=vocab_size)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print("✅ Transcriber Loaded.")
    return model, idx_to_gloss 

def run_transcriber(model, batch_tensor, x_lens, idx_to_gloss) -> list:
    """Passes the [1, T, 42, 10] tensor through the CTC model to get Glosses."""
    with torch.no_grad():
        # Forward pass (matches Stage 2 train loop)
        logits, out_lens = model(batch_tensor, x_lens) # logits shape: [1, max_tokens, Vocab_Size]
        
    # Decode CTC (argmax -> remove repeats -> remove blanks)
    preds = logits.argmax(dim=-1)[0].cpu().numpy() # Process the single item in batch
    valid_len = out_lens[0].item()
    
    decoded_glosses = []
    last_tok = BLANK_IDX
    
    for tok in preds[:valid_len]:
        if tok != BLANK_IDX and tok != last_tok:
            gloss = idx_to_gloss.get(tok, f"<UNK:{tok}>")
            decoded_glosses.append(gloss)
        last_tok = tok
        
    print(f"🗣️  Transcribed Glosses: {decoded_glosses}")
    return decoded_glosses

# =========================================================
# 3. The Stage 3 "Translator"
# =========================================================
def load_translator():
    """Loads the fine-tuned HuggingFace T5 Model."""
    print("⏳ Loading Stage 3 T5 Translator...")
    if not os.path.exists(STAGE3_DIR):
         raise FileNotFoundError(f"Cannot find T5 weights: {STAGE3_DIR}")
         
    tokenizer = AutoTokenizer.from_pretrained(STAGE3_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(STAGE3_DIR)
    model.to(DEVICE)
    model.eval()
    print("✅ Translator Loaded.")
    return model, tokenizer

def run_translator(gloss_list: list, model, tokenizer) -> str:
    """Translates the list of glosses into natural English."""
    if not gloss_list:
        return "[No Signs Detected]"
        
    # Join list and add prefix — must match train_stage_3.py exactly
    raw_string = " ".join(gloss_list)
    input_text = f"translate ASL gloss to English: {raw_string}"
    print(f"📝 T5 Input: '{input_text}'")
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=32, truncation=True)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=48,
            num_beams=4,
            early_stopping=True
        )
        
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    print("🚀 Starting Offline Pipeline Test...\n")
    
    # 1. Setup Models
    s2_model, idx_to_gloss = load_transcriber()
    s3_model, s3_tokenizer = load_translator()
    print("-" * 50)
    
    # 2. Define ~100 test cases from slt_stage3_dataset_final.csv (only glosses present in ASL_landmarks_float16)
    test_cases = [
        (['SYSTEM', 'I', 'NEED'], 'As for the system, I need it.'),
        (['NIGHT', 'MAN', 'GO', 'BANK'], 'At night, the man is going to the bank.'),
        (['TOMORROW', 'TEACHER', 'MEET', 'BABY'], 'Tomorrow, the teacher will meet the baby.'),
        (['YESTERDAY', 'TEAM', 'DELETE', 'PASSWORD'], 'Yesterday, the team deleted the password.'),
        (['COME', 'HOUSE'], 'Come to the house.'),
        (['THEY', 'NOT', 'WANT', 'FOOD'], 'They do not want some food.'),
        (['CODE', 'USE', 'I'], 'I am using the code.'),
        (['NOW', 'I', 'GO', 'CITY', 'SEE', 'IDEA'], 'Now, I am going to the city to see the idea.'),
        (['YESTERDAY', 'THEY', 'GO', 'PLACE', 'BUY', 'BOOK'], 'Yesterday, they went to the place and bought the book.'),
        (['PLEASE', 'HELP'], 'Please help.'),
        (['NOW', 'YOU', 'GO', 'HOUSE', 'MEET', 'WOMAN'], 'Now, you are going to the house to meet the woman.'),
        (['NOW', 'YOU', 'GO', 'HOUSE', 'SEE', 'EXAM'], 'Now, you are going to the house to see the exam.'),
        (['NIGHT', 'WORKER', 'SEE', 'IMPORTANT', 'BROTHER'], 'At night, the worker is seeing the important brother.'),
        (['TEACHER', 'NOT', 'LIKE', 'HOSPITAL'], 'The teacher does not like the hospital.'),
        (['COMPUTER', 'I', 'DEVELOP'], 'I am developing the computer.'),
        (['TOMORROW', 'I', 'GO', 'CITY', 'SEE', 'DESIGN'], 'Tomorrow, I will go to the city and see the design.'),
        (['PLEASE', 'GO'], 'Please go.'),
        (['I', 'WANT', 'TO', 'EAT', 'FOOD'], 'I want to eat food.'),
        (['AFTERNOON', 'LANGUAGE', 'THEY', 'GIVE'], 'In the afternoon, they are giving the language.'),
        (['YESTERDAY', 'I', 'GO', 'CITY', 'MEET', 'BROTHER'], 'Yesterday, I went to the city and met the brother.'),
        (['YESTERDAY', 'FRIEND', 'SEE', 'LOW', 'WORD'], 'Yesterday, the friend saw the low word.'),
        (['I', 'UNDERSTAND', 'COMPUTER'], 'I understand the computer.'),
        (['STUDENT', 'SEND', 'MONEY'], 'The student is sending the money.'),
        (['WORKER', 'CREATE', 'DATA'], 'The worker is creating the data.'),
        (['AFTERNOON', 'YOU', 'SEND', 'CODE'], 'In the afternoon, you are sending the code.'),
        (['CITY', 'GO', 'I'], 'I am going to the city.'),
        (['NOW', 'MOTHER', 'CREATE', 'NAME'], 'Now, the mother is creating the name.'),
        (['PLEASE', 'COME'], 'Please come.'),
        (['AFTERNOON', 'FRIEND', 'UPLOAD', 'CAMERA'], 'In the afternoon, the friend is uploading the camera.'),
        (['TEACHER', 'NOT', 'SEE', 'INTERNET'], 'The teacher does not see the internet.'),
        (['TODAY', 'I', 'GO', 'HOSPITAL', 'MEET', 'MAN'], 'Today, I am going to the hospital to meet the man.'),
        (['TODAY', 'THEY', 'SEE', 'GOOD', 'FAMILY'], 'Today, they are seeing the good family.'),
        (['TOMORROW', 'FAMILY', 'DOWNLOAD', 'PASSWORD'], 'Tomorrow, the family will download the password.'),
        (['POLICE', 'NOT', 'SEE', 'EXAM'], 'The police do not see the exam.'),
        (['TODAY', 'YOU', 'COME', 'COUNTRY'], 'Today, you are coming to the country.'),
        (['TOMORROW', 'POLICE', 'MOVE', 'MARKET'], 'Tomorrow, the police will move to the market.'),
        (['TEACHER', 'NOT', 'NEED', 'SCHOOL'], 'The teacher does not need the school.'),
        (['AFTERNOON', 'SISTER', 'SEND', 'PART'], 'In the afternoon, the sister is sending the part.'),
        (['AFTERNOON', 'POLICE', 'DELETE', 'MODEL'], 'In the afternoon, the police are deleting the model.'),
        (['NIGHT', 'WORKER', 'DRIVE', 'CAR'], 'At night, the worker is driving the car.'),
        (['HOW', 'MANY', 'MONEY'], 'How much money?'),
        (['I', 'FORGET', 'SCHOOL'], 'I forget the school.'),
        (['NIGHT', 'HE', 'RUN', 'ROOM'], 'At night, he is running to the room.'),
        (['AFTERNOON', 'FRIEND', 'DEVELOP', 'VIDEO'], 'In the afternoon, the friend is developing the video.'),
        (['GROUP', 'NOT', 'NEED', 'COUNTRY'], 'The group do not need the country.'),
        (['MORNING', 'BOSS', 'SEE', 'IMPORTANT', 'SISTER'], 'In the morning, the boss is seeing the important sister.'),
        (['POLICE', 'UPLOAD', 'DATA'], 'The police are uploading the data.'),
        (['YESTERDAY', 'YOU', 'GO', 'ROOM', 'SEE', 'QUESTION'], 'Yesterday, you went to the room and saw the question.'),
        (['TOMORROW', 'YOU', 'GO', 'CITY', 'BUY', 'VIDEO'], 'Tomorrow, you will go to the city and buy the video.'),
        (['TOMORROW', 'GROUP', 'SEE', 'EASY', 'BUS'], 'Tomorrow, the group will see the easy bus.'),
        (['FATHER', 'NOT', 'SEE', 'PLACE'], 'The father does not see the place.'),
        (['NIGHT', 'FRIEND', 'SEE', 'SIMPLE', 'SOLUTION'], 'At night, the friend is seeing the simple solution.'),
        (['HOW', 'MANY', 'BANK'], 'How many banks?'),
        (['MY', 'STUDENT', 'WHERE'], 'Where is my student?'),
        (['TODAY', 'BROTHER', 'SEE', 'IMPORTANT', 'LANGUAGE'], 'Today, the brother is seeing the important language.'),
        (['TODAY', 'GROUP', 'WANT', 'QUESTION'], 'Today, the group want the question.'),
        (['MOTHER', 'PAY', 'CAMERA'], 'The mother is paying the camera.'),
        (['TODAY', 'DOCTOR', 'FEEL', 'BUSY'], 'Today, the doctor is feeling busy.'),
        (['HELLO', 'HOW', 'YOU'], 'Hello, how are you?'),
        (['COME', 'OFFICE'], 'Come to the office.'),
        (['THEY', 'LIKE', 'BANK'], 'They like the bank.'),
        (['PLEASE', 'GIVE', 'INTERNET'], 'Give me the internet, please.'),
        (['THEY', 'REMEMBER', 'COUNTRY'], 'They remember the country.'),
        (['AFTERNOON', 'COMPUTER', 'DEVELOP', 'THEY'], 'In the afternoon, they are developing the computer.'),
        (['NIGHT', 'TEACHER', 'LEARN', 'PASSWORD'], 'At night, the teacher is learning the password.'),
        (['TOMORROW', 'TEACHER', 'SEE', 'SMALL', 'MARKET'], 'Tomorrow, the teacher will see the small market.'),
        (['WORKER', 'READ', 'SYSTEM'], 'The worker is reading the system.'),
        (['NIGHT', 'POLICE', 'WRITE', 'MONEY'], 'At night, the police are writing the money.'),
        (['HOW', 'MANY', 'ROOM'], 'How many rooms?'),
        (['PHONE', 'HE', 'SEND'], 'He is sending the phone.'),
        (['NIGHT', 'BROTHER', 'SEE', 'SMALL', 'PROBLEM'], 'At night, the brother is seeing the small problem.'),
        (['TEAM', 'DELETE', 'PASSWORD'], 'The team are deleting the password.'),
        (['WOMAN', 'NOT', 'SEE', 'SENTENCE'], 'The woman does not see the sentence.'),
        (['YESTERDAY', 'THEY', 'GO', 'MARKET', 'SEE', 'CAMERA'], 'Yesterday, they went to the market and saw the camera.'),
        (['TOMORROW', 'I', 'GO', 'ROOM', 'SEE', 'DESIGN'], 'Tomorrow, I will go to the room and see the design.'),
        (['HOW', 'MANY', 'PROBLEM'], 'How many problems?'),
        (['HOW', 'MANY', 'MEETING'], 'How many meetings?'),
        (['NIGHT', 'POLICE', 'FEEL', 'ANGRY'], 'At night, the police are feeling angry.'),
        (['NOW', 'FAMILY', 'FEEL', 'SAD'], 'Now, the family are feeling sad.'),
        (['CHILD', 'NOT', 'SEE', 'LANGUAGE'], 'The child does not see the language.'),
        (['TOMORROW', 'FAMILY', 'EAT', 'FOOD'], 'Tomorrow, the family will eat some food.'),
        (['TODAY', 'WOMAN', 'DRINK', 'WATER'], 'Today, the woman is drinking some water.'),
        (['TOMORROW', 'FAMILY', 'FEEL', 'READY'], 'Tomorrow, the family will feel ready.'),
        (['YESTERDAY', 'WORKER', 'COME', 'ROAD'], 'Yesterday, the worker came to the road.'),
        (['PLEASE', 'GIVE', 'SYSTEM'], 'Give me the system, please.'),
        (['STOP', 'WAIT'], 'Stop waiting.'),
        (['SORRY', 'MY', 'BAD'], 'Sorry, my bad.'),
        (['I', 'LIKE', 'STUDENT'], 'I like the student.'),
        (['TEACHER', 'FIX', 'EXAM'], 'The teacher is fixing the exam.'),
        (['NOW', 'I', 'RUN', 'PLACE'], 'Now, I am running to the place.'),
        (['MEETING', 'READ', 'YOU'], 'You are reading the meeting.'),
        (['COME', 'BANK'], 'Come to the bank.'),
        (['MORNING', 'POLICE', 'LEARN', 'PASSWORD'], 'In the morning, the police are learning the password.'),
        (['TEACHER', 'NOT', 'UNDERSTAND', 'BANK'], 'The teacher does not understand the bank.'),
        (['HE', 'LOVE', 'CAMERA'], 'He loves the camera.'),
        (['THEY', 'DEVELOP', 'CODE'], 'They are developing the code.'),
        (['PLEASE', 'GIVE', 'MONEY'], 'Give me the money, please.'),
        (['NIGHT', 'GROUP', 'UPLOAD', 'COMPUTER'], 'At night, the group are uploading the computer.'),
        (['TODAY', 'PROBLEM', 'YOU', 'BUY'], 'Today, you are buying the problem.'),
        (['YESTERDAY', 'TEACHER', 'DELETE', 'CODE'], 'Yesterday, the teacher deleted the code.'),
    ]
    
    correct = 0
    for i, (sentence_glosses, expected) in enumerate(test_cases, 1):
        print(f"\n📌 Test case {i}/{len(test_cases)}: {' '.join(sentence_glosses)}")
        print(f"   Expected: {expected}")
        try:
            sentence_files = [resolve_gloss_to_file(g, ASL_DATA_DIR) for g in sentence_glosses]
            batch_tensor, x_lens = load_and_concatenate_signs(sentence_files, ASL_DATA_DIR)
            gloss_sequence = run_transcriber(s2_model, batch_tensor, x_lens, idx_to_gloss)
            final_english = run_translator(gloss_sequence, s3_model, s3_tokenizer)
            final_english = final_english.strip()
            is_correct = _normalize_for_compare(final_english) == _normalize_for_compare(expected)
            if is_correct:
                correct += 1
            print("-" * 50)
            print(f"   Got:      {final_english}")
            print(f"   {'✅ CORRECT' if is_correct else '❌ WRONG'}")
            print("-" * 50)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("Skipping this test case. Ensure your .npy files are in the correct directory.")
    
    print("\n" + "=" * 50)
    print(f"📊 RESULT: {correct}/{len(test_cases)} test cases correct")
    print("=" * 50)