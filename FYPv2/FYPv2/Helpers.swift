//
//  Helpers.swift
//  FYPv2
//
//  Created by 武子健 on 21/3/2021.
//
import Foundation
import UIKit
import CoreML
import Accelerate

// The labels for the 80 classes.
let labels:[[String]] = [
    ["person", "170"],
    ["bicycle", "90"],
    ["car", "160"],
    ["motorbike", "100"],
    ["airplane", "600"],
    ["bus", "300"],
    ["train", "300"],
    ["truck", "300"],
    ["boat", "600"],
    ["traffic light", "100"],
    ["Rubish Bin", "60"],
    ["stop sign", "100"],
    ["parking meter", "100"],
    ["bench", "100"],
    ["bird", "30"],
    ["cat", "30"],
    ["dog", "30"],
    ["horse", "200"],
    ["sheep", "100"],
    ["cow", "100"],
    ["elephant", "300"],
    ["bear", "200"],
    ["zebra", "200"],
    ["giraffe", "300"],
    ["backpack", "50"],
    ["umbrella", "100"],
    ["handbag", "50"],
    ["tie", "50"],
    ["suitcase", "50"],
    ["frisbee", "50"],
    ["skis", "50"],
    ["snowboard", "100"],
    ["sports ball", "30"],
    ["kite", "30"],
    ["baseball bat", "50"],
    ["baseball glove", "10"],
    ["skateboard",  "100"],
    ["surfboard", "100"],
    ["tennis racket", "80"],
    ["bottle", "30"],
    ["wine glass", "40"],
    ["cup", "10"],
    ["fork", "10"],
    ["knife", "20"],
    ["spoon", "10"],
    ["bowl", "10"],
    ["banana", "20"],
    ["apple", "5"],
    ["sandwich", "5"],
    ["orange", "5"],
    ["broccoli", "5"],
    ["carrot", "5"],
    ["hot dog", "5"],
    ["pizza", "10"],
    ["donut", "5"],
    ["cake", "5"],
    ["chair", "100"],
    ["sofa", "100"],
    ["pottedplant", "50"],
    ["bed", "50"],
    ["diningtable", "100"],
    ["toilet", "200"],
    ["tvmonitor", "40"],
    ["laptop", "30"],
    ["mouse", "5"],
    ["remote", "5"],
    ["keyboard", "5"],
    ["cell phone", "5"],
    ["microwave", "30"],
    ["oven", "30"],
    ["toaster", "30"],
    ["sink", "50"],
    ["refrigerator", "200"],
    ["book", "5"],
    ["clock", "5"],
    ["vase", "5"],
    ["scissors", "10"],
    ["teddy bear", "50"],
    ["hair drier", "40"],
    ["toothbrush", "10"]
]

// calculate the height of the object and
//let ObjectHeight = [170, 100, 170, 100]


// anchor boxes
let anchors: [Float] = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

/**
  Removes bounding boxes that overlap too much with other boxes that have
  a higher score.

  Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc

  - Parameters:
    - boxes: an array of bounding boxes and their scores
    - limit: the maximum number of boxes that will be selected
    - threshold: used to decide whether boxes overlap too much
*/
func nonMaxSuppression(boxes: [YOLO.Prediction], limit: Int, threshold: Float) -> [YOLO.Prediction] {

  // Do an argsort on the confidence scores, from high to low.
  let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }

  var selected: [YOLO.Prediction] = []
  var active = [Bool](repeating: true, count: boxes.count)
  var numActive = active.count

  // The algorithm is simple: Start with the box that has the highest score.
  // Remove any remaining boxes that overlap it more than the given threshold
  // amount. If there are any boxes left (i.e. these did not overlap with any
  // previous boxes), then repeat this procedure, until no more boxes remain
  // or the limit has been reached.
  outer: for i in 0..<boxes.count {
    if active[i] {
      let boxA = boxes[sortedIndices[i]]
      selected.append(boxA)
      if selected.count >= limit { break }

      for j in i+1..<boxes.count {
        if active[j] {
          let boxB = boxes[sortedIndices[j]]
          if IOU(a: boxA.rect, b: boxB.rect) > threshold {
            active[j] = false
            numActive -= 1
            if numActive <= 0 { break outer }
          }
        }
      }
    }
  }
  return selected
}

/**
  Computes intersection-over-union overlap between two bounding boxes.
*/
public func IOU(a: CGRect, b: CGRect) -> Float {
  let areaA = a.width * a.height
  if areaA <= 0 { return 0 }

  let areaB = b.width * b.height
  if areaB <= 0 { return 0 }

  let intersectionMinX = max(a.minX, b.minX)
  let intersectionMinY = max(a.minY, b.minY)
  let intersectionMaxX = min(a.maxX, b.maxX)
  let intersectionMaxY = min(a.maxY, b.maxY)
  let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                         max(intersectionMaxX - intersectionMinX, 0)
  return Float(intersectionArea / (areaA + areaB - intersectionArea))
}

extension Array where Element: Comparable {
  /**
    Returns the index and value of the largest element in the array.
  */
  public func argmax() -> (Int, Element) {
    precondition(self.count > 0)
    var maxIndex = 0
    var maxValue = self[0]
    for i in 1..<self.count {
      if self[i] > maxValue {
        maxValue = self[i]
        maxIndex = i
      }
    }
    return (maxIndex, maxValue)
  }
}

/**
  Logistic sigmoid.
*/
public func sigmoid(_ x: Float) -> Float {
  return 1 / (1 + exp(-x))
}

/**
  Computes the "softmax" function over an array.

  Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/

  This is what softmax looks like in "pseudocode" (actually using Python
  and numpy):

      x -= np.max(x)
      exp_scores = np.exp(x)
      softmax = exp_scores / np.sum(exp_scores)

  First we shift the values of x so that the highest value in the array is 0.
  This ensures numerical stability with the exponents, so they don't blow up.
*/
public func softmax(_ x: [Float]) -> [Float] {
  var x = x
  let len = vDSP_Length(x.count)

  // Find the maximum value in the input array.
  var max: Float = 0
  vDSP_maxv(x, 1, &max, len)

  // Subtract the maximum from all the elements in the array.
  // Now the highest value in the array is 0.
  max = -max
  vDSP_vsadd(x, 1, &max, &x, 1, len)

  // Exponentiate all the elements in the array.
  var count = Int32(x.count)
  vvexpf(&x, x, &count)

  // Compute the sum of all exponentiated values.
  var sum: Float = 0
  vDSP_sve(x, 1, &sum, len)

  // Divide each element by the sum. This normalizes the array contents
  // so that they all add up to 1.
  vDSP_vsdiv(x, 1, &sum, &x, 1, len)

  return x
}
