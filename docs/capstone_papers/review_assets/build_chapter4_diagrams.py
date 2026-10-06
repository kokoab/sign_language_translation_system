"""Render Chapter 4 design diagrams; no measured results are generated."""
from build_review_assets import flow, save, plt, BLUE, TEAL
from matplotlib.patches import Rectangle, Ellipse, Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath


def deployment_flow():
    flow('deployment_flow', 'Model deployment and ATLAS workflow', {
        'trained': (6,23,'Trained boundary, recognition, visual and English models','box'),
        'export': (6,21,'Model export and application loading','box'),
        'camera': (6,19,'Device camera frames','box'),
        'vision': (3,17,'Landmark extractor: movement features','box'),
        'hands': (10,17,'MobileCLIP2 hand-image features','box'),
        'boundary': (3,14.5,'Boundary model: estimate candidate sign intervals','output'),
        'recognition': (6,12,'Squeezeformer recognizer: score candidate signs','output'),
        'decoder': (6,9.5,'Segmental decoder: select and commit stable glosses','box'),
        'gloss': (6,7.5,'Gloss Management: ordered sequence','box'),
        'english': (6,5.5,'T5-efficient-tiny: incremental English','box'),
        'finish': (6,3.7,'Finish button or held open palms','box'),
        'final': (6,1.9,'Finalize English text','box'),
        'output': (6,.3,'English text and local speech','output'),
    }, [('trained','export',''),('export','camera',''),
        ('camera','vision',''),('camera','hands',''),('vision','boundary',''),
        ('boundary','recognition','Candidate landmark intervals'),
        ('hands','recognition','Appearance features'),
        ('recognition','decoder','Sign scores'),('decoder','gloss','Accepted glosses'),
        ('gloss','english',''),('english','finish',''),('finish','final',''),
        ('final','output','')], (12,16), (13,24))


def main():
    deployment_flow()
    context_and_data_flow()
    use_cases()
    sashimi()


def context_and_data_flow():
    flow('context', 'ATLAS system context', {
        'camera': (2, 6, 'Device camera: signing input', 'box'),
        'system': (7, 6, 'ATLAS application', 'output'),
        'user': (7, 9, 'Application user: navigation and Finish controls', 'box'),
        'text': (12, 7.5, 'English text and recognized glosses', 'output'),
        'speech': (12, 4.5, 'Local speech output', 'output'),
        'history': (7, 2.5, 'Saved session history', 'box'),
    }, [('camera','system','Video'), ('user','system','Controls'),
        ('system','text',''), ('system','speech',''), ('system','history','Session record')], (12,8), (14,10))
    flow('data_flow', 'ATLAS data flow', {
        'input': (5, 15, 'Camera frames', 'box'),
        'visual': (5, 13, '1. Visual Extraction: landmarks and hand-image features', 'box'),
        'recognition': (5, 10.5, '2. Streaming Sign Recognition', 'box'),
        'gloss': (5, 8, '3. Gloss Management', 'box'),
        'english': (5, 6, '4. T5-efficient-tiny: incremental English generation', 'box'),
        'finish': (5, 4.3, 'Finish button or held open palms', 'box'),
        'finalize': (5, 2.6, 'Finalize English text', 'box'),
        'output': (5, .9, 'English text and local speech', 'output'),
        'history': (10, .9, 'Session history', 'box'),
    }, [('input','visual','Video'), ('visual','recognition','Landmarks and image features'),
        ('recognition','gloss','Accepted glosses'), ('gloss','english','Ordered gloss sequence'),
        ('english','finish',''), ('finish','finalize','Completion control'),
        ('finalize','output','English text'),
        ('output','history','Session record')], (11,13), (12,16))


def use_cases():
    fig, ax = plt.subplots(figsize=(11,8))
    ax.set(xlim=(0,12), ylim=(0,10)); ax.axis('off')
    ax.add_patch(Rectangle((4,.5),7.5,9, fill=False, edgecolor=BLUE, linewidth=1.4))
    ax.text(7.75,9.1,'ATLAS',ha='center',weight='bold')
    ax.add_patch(Circle((1.5,5.7),.28,fill=False,edgecolor=BLUE))
    ax.plot([1.5,1.5],[5.4,4.5], color=BLUE)
    ax.plot([.95,2.05],[5.05,5.05],color=BLUE)
    ax.plot([.95,1.5,2.05],[3.9,4.5,3.9],color=BLUE)
    ax.text(1.5,3.45,'Non-Sign Language\nUser',ha='center',va='top')
    labels=['Access Live recognition','Finish and receive English\ntext and speech','Browse Glosses and\nview demonstrations','Practice selected or random signs','Review session History']
    for y,label in zip([8,6.5,5,3.5,2],labels):
        ax.add_patch(Ellipse((8,y),5.6,1.05,fc='#e8f1f7',ec=BLUE))
        ax.text(8,y,label,ha='center',va='center',fontsize=10)
        ax.plot([2.1,5.2],[5.05,y],color='#718096',linewidth=.9)
    fig.suptitle('ATLAS application use cases',weight='bold',fontsize=14)
    save(fig,'use_cases')


def sashimi():
    """Reproduce the author-supplied five-phase layout as editable artwork."""
    fig, ax = plt.subplots(figsize=(10.48, 7.24))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0,1048), ylim=(724,0)); ax.axis('off')
    labels = ['PLANNING', 'DESIGNING', 'DEVELOPMENT', 'TESTING', 'IMPLEMENTATION']
    colors = ['#403487', '#104b7e', '#075747', '#683c04', '#742d15']
    positions = [(46,48), (207,185), (368,322), (529,459), (690,596)]
    for i, ((x,y), label, color) in enumerate(zip(positions,labels,colors)):
        ax.add_patch(FancyBboxPatch((x,y),321,102,
                     boxstyle='round,pad=0,rounding_size=27',
                     linewidth=.7,edgecolor=color,facecolor=color,zorder=2))
        ax.text(x+160.5,y+51,label,ha='center',va='center',
                fontsize=16,weight='bold',color='white',zorder=3)
        if i == 4:
            continue
        # Forward progression around the right side; feedback around the left.
        nx,ny=positions[i+1]
        routes = [
            [(x+321,y+51),(x+344,y+51),(x+369,y+51),(x+369,y+77),(x+369,ny)],
            [(nx,ny+51),(nx-24,ny+51),(nx-49,ny+51),(nx-49,ny+25),(nx-49,y+102)],
        ]
        for points in routes:
            path=MplPath(points,[MplPath.MOVETO,MplPath.LINETO,
                                MplPath.CURVE3,MplPath.CURVE3,MplPath.LINETO])
            ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',
                          mutation_scale=17,lw=2.4,color='#414440',zorder=1))
    save(fig,'sashimi',tight=False)

if __name__=='__main__':
    main()
