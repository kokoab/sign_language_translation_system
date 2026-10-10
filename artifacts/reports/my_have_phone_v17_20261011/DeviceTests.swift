extension RunnerTests {
  func testMYHaveDeviceProfile() throws {
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let dir = docs.appendingPathComponent("my_have_phone_20261011")
    func read(_ name: String) throws -> Any { try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent(name))) }
    let configuration = try read("configuration.json") as! [String: Any]
    let variant = configuration["variant"] as! String
    var records: [[String: Any]] = []
    func footprint() -> Double {
      var info = task_vm_info_data_t()
      var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
      let status = withUnsafeMutablePointer(to: &info) { p in
        p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
          task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
      }
      return status == KERN_SUCCESS ? Double(info.phys_footprint) / 1048576 : -1
    }
    func stats(_ a: [Double]) -> [String: Any] {
      let s = a.sorted()
      guard !s.isEmpty else { return [:] }
      return ["frames": s.count, "median_ms": s[s.count/2], "p90_ms": s[Int(Double(s.count)*0.9)],
        "p99_ms": s[Int(Double(s.count)*0.99)], "max_ms": s.last!, "mean_ms": s.reduce(0,+)/Double(s.count),
        "over50ms": s.filter{$0 > 50}.count, "samples_ms": a]
    }
    var metadata: [String: Any] = ["variant":variant, "device":UIDevice.current.model,
      "os":UIDevice.current.systemVersion,"low_power":ProcessInfo.processInfo.isLowPowerModeEnabled,
      "thermal_start":ProcessInfo.processInfo.thermalState.rawValue,"before_load_mb":footprint()]
    LiveHandEncoder.packageName = "MobileCLIP2S0ImageEncoderV17FP16"
    let loadStart = CACurrentMediaTime()
    let engine = try LiveReelEngine(recognizerUnits: .all, encoderUnits: .all)
    metadata["load_ms"] = (CACurrentMediaTime()-loadStart)*1000
    metadata["loaded_mb"] = footprint()
    let warmStart = CACurrentMediaTime()
    try engine.warm()
    metadata["warm_ms"] = (CACurrentMediaTime()-warmStart)*1000
    metadata["warmed_mb"] = footprint()
    func save() throws {
      metadata["thermal_end"] = ProcessInfo.processInfo.thermalState.rawValue
      metadata["end_mb"] = footprint()
      try JSONSerialization.data(withJSONObject:["metadata":metadata,"records":records], options:[.sortedKeys])
        .write(to: docs.appendingPathComponent("my_have_" + variant + ".json"), options:.atomic)
    }
    let cases = try read("cases.json") as! [[String: Any]]
    for pass in 0..<2 {
      for item in (pass == 0 ? cases : Array(cases.reversed())) {
        try autoreleasepool {
          let trace = try read(item["path"] as! String) as! [String: Any]
          let frames = (trace["frames"] as! [[String: Any]]).filter{$0["start"] == nil && ($0["held"] as? Bool) != true}
          func points(_ r:[String:Any], _ k:String) -> [LivePoint] {
            (r[k] as! [[Double]]).map{LivePoint(x:Float($0[0]),y:Float($0[1]),c:Float($0[2]))}
          }
          func hand(_ r:[String:Any], _ k:String) -> LiveHand? {
            guard r[k] != nil else {return nil}
            var h=LiveHand();h.points=points(r,k);h.chirality=r[k+"c"] as? String ?? "unknown";h.score=Float(r[k+"s"] as? Double ?? 0);return h
          }
          let observations = frames.map {r in LiveObservation(seconds:r["s"] as! Double,width:r["w"] as! Int,height:r["h"] as! Int,
            left:hand(r,"l"),right:hand(r,"r"),body:points(r,"b"),face:points(r,"f"),faceForFeatures:r["ff"] as! Bool)}
          let hands = frames.map {r -> LiveHandFrame in
            var h=LiveHandFrame()
            if let encoded=r["ef32"] as? String { h.embeddings=Data(base64Encoded:encoded)!.withUnsafeBytes{Array($0.bindMemory(to:Float.self))} }
            else {h.embeddings=Data(base64Encoded:r["e"] as! String)!.withUnsafeBytes{$0.bindMemory(to:Float16.self).map(Float.init)}}
            h.valid=(r["v"] as! [Double]).map(Float.init);h.boxes=(r["x"] as! [Double]).map(Float.init);return h
          }
          engine.reset();let runtime=engine.wordsGated
          var words:[LiveWord]=[], times:[Double]=[], publications:[[String:Any]]=[]
          var peak=footprint()
          let thermalBefore=ProcessInfo.processInfo.thermalState.rawValue
          for i in observations.indices {
            let start=CACurrentMediaTime()
            let emitted=try autoreleasepool {try runtime.observe(observations[i],image:nil,hand:hands[i])}
            times.append((CACurrentMediaTime()-start)*1000);words += emitted
            publications += emitted.map{["gloss":$0.gloss,"start_frame":$0.startFrame,"identity_seconds":$0.commitSeconds,
              "published_input_seconds":observations[i].seconds,"published_frame":i,"finish":false]}
            if i % 20 == 0 {peak=max(peak,footprint())}
          }
          let flushStart=CACurrentMediaTime();let tail=try runtime.finish();let flushMS=(CACurrentMediaTime()-flushStart)*1000
          words += tail
          publications += tail.map{["gloss":$0.gloss,"start_frame":$0.startFrame,"identity_seconds":$0.commitSeconds,"finish":true]}
          let visible=words.filter{!$0.gloss.hasPrefix("FS_")}
          records.append(["kind":"trace","id":item["id"]!,"pass":pass,"hypothesis":visible.map(\.gloss),"words":visible.map(\.json),
            "publications":publications,"timing":stats(times),"finish_ms":flushMS,"peak_sampled_mb":peak,
            "thermal_before":thermalBefore,"thermal_after":ProcessInfo.processInfo.thermalState.rawValue])
          print("MYHAVE trace \(variant) \(pass) \(item["id"]!) \(visible.map(\.gloss))")
        }
        try save()
      }
    }
    // Full engine replay uses actual presentation timestamps; timing excludes file decode and resize.
    for pass in 0..<2 {
      try autoreleasepool {
        engine.reset()
        let asset=AVURLAsset(url:dir.appendingPathComponent("take03.mp4"))
        let sem=DispatchSemaphore(value:0);var track:AVAssetTrack?;var transform=CGAffineTransform.identity
        Task {track=try? await asset.loadTracks(withMediaType:.video).first
          if let track {transform=(try? await track.load(.preferredTransform)) ?? .identity};sem.signal()}
        sem.wait();XCTAssertTrue(transform.isIdentity,"recording must be upright")
        let reader=try AVAssetReader(asset:asset)
        let output=AVAssetReaderTrackOutput(track:try XCTUnwrap(track),outputSettings:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData=false;reader.add(output);XCTAssertTrue(reader.startReading())
        let context=CIContext();var first:Double?;var deadline=0.0;var words:[LiveWord]=[],times:[Double]=[],modes:[[String:Any]]=[],publications:[[String:Any]]=[]
        var peak=footprint();let thermalBefore=ProcessInfo.processInfo.thermalState.rawValue
        while let sample=output.copyNextSampleBuffer(),let raw=CMSampleBufferGetImageBuffer(sample) {
          let pts=CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample));if first == nil {first=pts}
          let seconds=pts-first!;if seconds+1e-6 < deadline {continue};deadline=max(deadline+0.05,seconds)
          try autoreleasepool {
            var frame=raw;let w=CVPixelBufferGetWidth(raw),h=CVPixelBufferGetHeight(raw)
            if max(w,h)>LiveReelEngine.maximumSide {
              let scale=Double(LiveReelEngine.maximumSide)/Double(max(w,h));let tw=Int((Double(w)*scale).rounded()),th=Int((Double(h)*scale).rounded())
              var value:CVPixelBuffer?;CVPixelBufferCreate(kCFAllocatorDefault,tw,th,kCVPixelFormatType_32BGRA,nil,&value)
              frame=try XCTUnwrap(value);context.render(CIImage(cvPixelBuffer:raw).transformed(by:CGAffineTransform(scaleX:Double(tw)/Double(w),y:Double(th)/Double(h))),to:frame)
            }
            let start=CACurrentMediaTime();let step=try engine.process(frame,seconds:seconds,allowFinishGesture:true)
            times.append((CACurrentMediaTime()-start)*1000);words += step.committed
            publications += step.committed.map{["gloss":$0.gloss,"identity_seconds":$0.commitSeconds,"published_input_seconds":seconds]}
            if let event=step.spellEvent {modes.append(["seconds":seconds,"event":event])}
            if times.count % 20 == 0 {peak=max(peak,footprint())}
          }
        }
        XCTAssertEqual(reader.status,.completed)
        words += try engine.finish()
        records.append(["kind":"video","id":"take03","pass":pass,"hypothesis":words.map(\.gloss),"words":words.map(\.json),
          "spell_events":modes,"publications":publications,"timing":stats(times),"peak_sampled_mb":peak,
          "thermal_before":thermalBefore,"thermal_after":ProcessInfo.processInfo.thermalState.rawValue])
        print("MYHAVE video \(variant) \(pass) \(words.map(\.gloss))")
      }
      try save()
    }
    print("MYHAVE_COMPLETE \(variant) \(records.count)")
  }
}
