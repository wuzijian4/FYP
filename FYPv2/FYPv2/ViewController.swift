//
//  ViewController.swift
//  FYPv2
//
//  Created by 武子健 on 21/3/2021.
//
import UIKit
import Vision
import AVFoundation
import CoreMedia
import VideoToolbox
import SceneKit
import ModelIO

class ViewController: UIViewController {
    @IBOutlet weak var VideoPreview: UIView!
    
    @IBOutlet weak var timeLabel: UILabel!
    @IBOutlet weak var ClassificationView: UIVisualEffectView!
    @IBOutlet weak var Distance: UILabel!
    
    let yolo = YOLO()

    var videoCapture: VideoCapture!
    var request: VNCoreMLRequest!
    var startTimes: [CFTimeInterval] = []

    var boundingBoxes = [BoundingBox]()
    var colors: [UIColor] = []

    var framesDone = 0
    var frameCapturingStartTime = CACurrentMediaTime()
    let semaphore = DispatchSemaphore(value: 2)


    // focal length of camera
    var focalLength = Double(MDLCamera().focalLength)
    
    // height of screen in pixels format
    var pxHeight :Double = 2532

    // length of screen in pixels format
    var pxLength:Double = 1170
    
    // Height of screen in millimeters
    var picHeight:Double = 146.0
    
    // lenght of screen in millimeters
    var picLength:Double = 71.6
    
    // cache the closest object
    var ClosetObject : String  = "";
    
    // cache the occured time of the closet object
    var timeStamp : CLongLong = 0;
    
    // Audio notification function
    var synthesizer = AVSpeechSynthesizer()
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        timeLabel.text = ""
        Distance.text = ""
        Distance.lineBreakMode = NSLineBreakMode.byWordWrapping
        Distance.numberOfLines = 2

        self.VideoPreview.isHidden = false;
        self.VideoPreview.contentMode = .scaleAspectFit;
        setUpBoundingBoxes()
        setUpVision()
        setUpCamera()

        frameCapturingStartTime = CACurrentMediaTime()
    }

    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        print(#function)
    }

    // MARK: - Initialization

    func setUpBoundingBoxes() {
        for _ in 0..<YOLO.maxBoundingBoxes {
            boundingBoxes.append(BoundingBox())
        }

        // Make colors for the bounding boxes. There is one color for each class,
        // 80 classes in total.
        for r: CGFloat in [0.2, 0.4, 0.6, 0.85, 1.0] {
            for g: CGFloat in [0.6, 0.7, 0.8, 0.9] {
                for b: CGFloat in [0.6, 0.7, 0.8, 1.0] {
                    let color = UIColor(red: r, green: g, blue: b, alpha: 1)
                    colors.append(color)
                }
            }
        }
    }
  
    func setUpVision() {
        guard let visionModel = try? VNCoreMLModel(for: yolo.model.model) else {
            print("Error: could not create Vision model")
            return
        }

        request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)

        // NOTE: If you choose another crop/scale option, then you must also
        // change how the BoundingBox objects get scaled when they are drawn.
        // Currently they assume the full input image is used.
        request.imageCropAndScaleOption = .scaleFill
    }

    // AVCaptureSession.Preset.vga640x480
    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 15
        videoCapture.setUp(sessionPreset: AVCaptureSession.Preset.vga640x480) { success in
            if success {
                // Add the video preview into the UI.
                if let previewLayer = self.videoCapture.previewLayer {
                    self.VideoPreview.layer.addSublayer(previewLayer)
                    self.resizePreviewLayer()
                }

                // Add the bounding box layers to the UI, on top of the video preview.
                for box in self.boundingBoxes {
                    box.addToLayer(self.VideoPreview.layer)
                }

                // Once everything is set up, we can start capturing live video.
                self.videoCapture.start()
            }
        }
    }

    // MARK: - UI stuff

    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        resizePreviewLayer()
    }

    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }

    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = VideoPreview.layer.bounds
    }

    // MARK: - Doing inference
    func predictUsingVision(pixelBuffer: CVPixelBuffer) {
        // Measure how long it takes to predict a single video frame. Note that
        // predict() can be called on the next frame while the previous one is
        // still being processed. Hence the need to queue up the start times.
        startTimes.append(CACurrentMediaTime())

        // Vision will automatically resize the input image.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
    }

    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
           let features = observations.first?.featureValue.multiArrayValue {

            let boundingBoxes = yolo.computeBoundingBoxes(features: features)
            let elapsed = CACurrentMediaTime() - startTimes.remove(at: 0)
            showOnMainThread(boundingBoxes, elapsed)
        }
    }

    func showOnMainThread(_ boundingBoxes: [YOLO.Prediction], _ elapsed: CFTimeInterval) {
        DispatchQueue.main.async {

            self.show(predictions: boundingBoxes)

            let fps = self.measureFPS()
            self.timeLabel.text = String(format: "Elapsed %.5f seconds - %.2f FPS", elapsed, fps)

            self.semaphore.signal()
        }
    }

    func measureFPS() -> Double {
        // Measure how many frames were actually delivered per second.
        framesDone += 1
        let frameCapturingElapsed = CACurrentMediaTime() - frameCapturingStartTime
        let currentFPSDelivered = Double(framesDone) / frameCapturingElapsed
        if frameCapturingElapsed > 1 {
            framesDone = 0
            frameCapturingStartTime = CACurrentMediaTime()
        }
        return currentFPSDelivered
    }

    func show(predictions: [YOLO.Prediction])
    {
        // 初始值
        var miniDis : Double = 5000
        var miniDisObject = ""
        
        for i in 0..<boundingBoxes.count {
          if i < predictions.count {
            let prediction = predictions[i]

            // The predicted bounding box is in the coordinate space of the input
            // image, which is a square image of 416x416 pixels. We want to show it
            // on the video preview, which is as wide as the screen and has a 4:3
            // aspect ratio. The video preview also may be letterboxed at the top
            // and bottom.
            let width = view.bounds.width
            let height = width * 4 / 3
            let scaleX = width / CGFloat(YOLO.inputWidth)
            let scaleY = height / CGFloat(YOLO.inputHeight)
            let top = (view.bounds.height - height) / 2

            // Translate and scale the rectangle to our own coordinate system.
            var rect = prediction.rect
            rect.origin.x *= scaleX
            rect.origin.y *= scaleY
            rect.origin.y += top
            rect.size.width *= scaleX
            rect.size.height *= scaleY

            // Show the bounding box.
            let label = String(format: "%@ %.1f", labels[prediction.classIndex][0], prediction.score * 100)
            let color = colors[prediction.classIndex]
            boundingBoxes[i].show(frame: rect, label: label, color: color)
            
            // calculate the distance to select the object which is closest to the client
            // 物距 = （1 + 物高 * 竖屏像素 / 竖屏高度 / 像素数）* 焦距
            let objectHeight = Double(labels[prediction.classIndex][1]) ?? 0
            let objectPix = Double(rect.size.height)
            
            // 单位是毫米，要换算成米
            var distance :Double = (1 + (objectHeight*10) / (picHeight * objectPix / pxHeight))
            distance *= focalLength
            distance /= 1000
            if(distance < miniDis)
            {
                miniDis = distance
                miniDisObject = labels[prediction.classIndex][0]
            }
            
          } else {
            boundingBoxes[i].hide()
          }
        }
        
        if(miniDis == 5000 && miniDisObject == "")
        {
            self.Distance.text = "Calculating Distance ....."
        }
        else
        {
            self.Distance.text = "The closest object is \(miniDisObject) the distance is \(miniDis)"
        }
        
        // 记录下最近物体，以及时间，当连续三秒出现这个物体的时候，就报警。

        if miniDisObject == ClosetObject
        {
            // 如果当前物体和缓存里的物体相等，就检查时间戳，如果时间相差大过三秒，就报警。
            // audio notify
            
            // 初始化timestamp
            if timeStamp == 0
            {
                timeStamp = Date().milliStamp;
            }
            
            if Date().milliStamp - timeStamp >= 4000 && miniDis <= 3.0
            {
                
                
                // audio notify
                let output : String = "Warning : You are wihtin \(ceil(miniDis)) meters from \(miniDisObject)"
                timeStamp = Date().milliStamp;
                speakText(voiceOutdata: output);

            }
        }
        else if(miniDisObject != "")
        {
            //更新物体和时间戳
            ClosetObject = miniDisObject;
            timeStamp = Date().milliStamp;
//            synthesizer.stopSpeaking(at: AVSpeechBoundary.immediate);
            //synthesizer.pauseSpeaking(at: AVSpeechBoundary.immediate);
        }
        
      }
    
    func speakText(voiceOutdata: String ) {
        do
        {
            try AVAudioSession.sharedInstance().setCategory(AVAudioSession.Category.playAndRecord)

            try AVAudioSession.sharedInstance().setActive(true, options: .notifyOthersOnDeactivation)
        }
        catch
        {
                print("audioSession properties weren't set because of an error.")
        }

        synthesizer.stopSpeaking(at: AVSpeechBoundary.immediate);
        let utterance = AVSpeechUtterance(string: voiceOutdata)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")

        utterance.rate = 0.5;
        utterance.pitchMultiplier = 1.0;
        utterance.postUtteranceDelay = 0.5;
        print(voiceOutdata)
        synthesizer.speak(utterance)
    }

    private func disableAVSession() {
        do {
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            print("audioSession properties weren't disable.")
        }
    }
}

extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
        // For debugging.
        //predict(image: UIImage(named: "dog416")!); return

        semaphore.wait()

        if let pixelBuffer = pixelBuffer {
            // For better throughput, perform the prediction on a background queue
            // instead of on the VideoCapture queue. We use the semaphore to block
            // the capture queue and drop frames when Core ML can't keep up.
            DispatchQueue.global().async {
                //self.predict(pixelBuffer: pixelBuffer)
                self.predictUsingVision(pixelBuffer: pixelBuffer)
            }
        }
    }
    
}

extension Date{
    var timestamp : Int {
        let timeInterval : TimeInterval = self.timeIntervalSince1970;
        let timeStamp : Int = Int(timeInterval);
        
        return timeStamp;
    }
    
    var milliStamp : CLongLong{
        let timeInterval : TimeInterval = self.timeIntervalSince1970;
        let milliseconds : CLongLong = CLongLong(round(timeInterval*1000));
        
        return milliseconds;
    }
}




