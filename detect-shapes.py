#!/usr/bin/python

import sys
import cv2
import math
import numpy
import json

SHAPE_TRIANGLE = 1
SHAPE_RECTANGLE = 2
SHAPE_PENTAGON = 3
SHAPE_HEXAGON = 4
SHAPE_HEPTAGON = 5
SHAPE_OCTAGON = 6
SHAPE_NONAGON = 7
SHAPE_DECAGON = 8
SHAPE_CIRCLE = 9

usage = "Usage: detect_shapes.py <image-file> <camera-intrinsics-file> <object-distance-from-camera-mm> [--show]"


def calculateAngle(a: numpy.ndarray, b: numpy.ndarray):
    cos = numpy.inner(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))
    rad = numpy.arccos(numpy.clip(cos, -1.0, 1.0))
    return numpy.rad2deg(rad)


def calculateLength(l: float, focalLength: float, distance: float):
    return distance * l / focalLength


def parseArgs(args):
    try:
        distance = float(args[3])
    except:
        print("Please provide correct input for distance in millimeters!",
              file=sys.stderr)
        print(usage, file=sys.stderr)
        exit()
    show = False
    if len(args) > 4:
        if args[4] == '--show':
            show = True
    return args[1], args[2], distance, show


if len(sys.argv) < 4:
    print("Please provide arguments!", file=sys.stderr)
    print(usage, file=sys.stderr)
    exit()


def getFocalLength(cameraIntrinsicsFileName: str):
    cameraIntrinsicsFile = open(cameraIntrinsicsFileName, 'r')
    cameraIntrinsics = json.loads(cameraIntrinsicsFile.read())
    cameraIntrinsicsFile.close()
    return (cameraIntrinsics['ffx'] + cameraIntrinsics['ffy']) / 2


def getContours(image: numpy.ndarray):
    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresoldImage = cv2.threshold(grayImage, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresoldImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def getRadius(circle: numpy.ndarray):
    _, _, w, h = cv2.boundingRect(circle)
    return (w + h)/4


def detectShape(contour: numpy.ndarray):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    shape = cv2.approxPolyDP(contour, epsilon, True)

    if len(shape) == 3:
        return SHAPE_TRIANGLE, shape
    elif len(shape) == 4:
        return SHAPE_RECTANGLE, shape
    elif len(shape) == 5:
        return SHAPE_PENTAGON, shape
    elif len(shape) == 6:
        return SHAPE_HEXAGON, shape
    elif len(shape) == 7:
        return SHAPE_HEPTAGON, shape
    elif len(shape) == 8:
        return SHAPE_OCTAGON, shape
    elif len(shape) == 9:
        return SHAPE_NONAGON, shape
    elif len(shape) == 10:
        return SHAPE_DECAGON, shape
    else:
        return SHAPE_CIRCLE, shape


def makeLabel(image, shape: numpy.ndarray, name: str):
    cv2.drawContours(image, [shape], 0, (0, 0, 255), 4)
    x, y, _, _ = cv2.boundingRect(shape)
    coords = (x, y - 4)
    colour = (0, 255, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, coords, font, 1, colour, 1)


imageFileName, cameraIntrinsicsFileName, distance, showOption = parseArgs(
    sys.argv)
focalLength = getFocalLength(cameraIntrinsicsFileName)
image = cv2.imread(imageFileName)
contours = getContours(image)

shapesAmount = len(contours) - 1
detectedShapes = []

for i, contour in enumerate(contours):
    if i == 0:
        continue

    shapeType, shape = detectShape(contour)

    if shapeType == SHAPE_TRIANGLE:
        detectedShapes.append("triangle")
        shapeName = "Triangle"

        vec1 = numpy.array([shape[0][0][0] - shape[1][0][0],
                            shape[0][0][1] - shape[1][0][1]])
        vec2 = numpy.array([shape[2][0][0] - shape[1][0][0],
                            shape[2][0][1] - shape[1][0][1]])
        triangleAngle = calculateAngle(vec1, vec2)

    elif shapeType == SHAPE_RECTANGLE:
        detectedShapes.append("rectangle")
        shapeName = "Rectangle"
        x1 = shape[0][0][0]
        y1 = shape[0][0][1]
        x2 = shape[1][0][0]
        y2 = shape[1][0][1]
        pixelLength = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        rectangleSide = calculateLength(pixelLength, focalLength, distance)

    elif shapeType == SHAPE_CIRCLE:
        detectedShapes.append("circle")
        shapeName = "Circle"
        circleRadius = calculateLength(getRadius(shape), focalLength, distance)

    elif shapeType == SHAPE_HEPTAGON:
        shapeName = "heptagon"
    elif shapeType == SHAPE_OCTAGON:
        shapeName = "octagon"
    elif shapeType == SHAPE_NONAGON:
        shapeName = "nonagon"
    elif shapeType == SHAPE_DECAGON:
        shapeName = "decagon"

    if showOption:
        makeLabel(image, shape, shapeName)

print("amount of shapes:", shapesAmount)
print("Detected shapes:", ", ".join(detectedShapes))
print("Rectangle side: %fmm" % (rectangleSide))
print("Circle radius: %fmm" % (circleRadius))
print("Triangle angle: %f°" % (triangleAngle))

if showOption:
    cv2.imshow("Detected shapes -" + ", ".join(detectedShapes), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
