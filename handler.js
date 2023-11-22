// handler.js
require('dotenv/config');
require('@tensorflow/tfjs-node');
const canvas = require('canvas');

const fs = require('fs');

const faceapi = require('@vladmandic/face-api');
const path = require('path');

const tmpFolder = path.resolve(__dirname, '.', 'tmp');

module.exports.detectFace = async (event) => {
  try {
    const MODEL_URL = './model';
    const { Canvas, Image, ImageData } = canvas;
    faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

    await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);
    const isFaceDetectionModelLoaded = faceapi.nets.ssdMobilenetv1.isLoaded;

    if (!isFaceDetectionModelLoaded) {
      throw new Error('Não carregou');
    }

    // export const faceDetectionNet = tinyFaceDetector

    // SsdMobilenetv1Options
    const minConfidence = 0.5;

    // TinyFaceDetectorOptions

    function getFaceDetectorOptions(net) {
      return new faceapi.SsdMobilenetv1Options({ minConfidence });
    }

    const faceDetectionOptions = getFaceDetectorOptions();
    const compareUrl = req?.body?.compareUrl;
    const avatarFilename = req?.file?.filename;
    const avatar_url = compareUrl;
    const originalPath = path.resolve(tmpFolder, avatarFilename);
    const referenceImage = await canvas.loadImage(avatar_url);

    const queryImage = await canvas.loadImage(originalPath);

    const resultsRef = await faceapi.detectAllFaces(
      referenceImage,
      faceDetectionOptions
    );
    const resultsQuery = await faceapi.detectAllFaces(
      queryImage,
      faceDetectionOptions
    );

    const faceImages1 = await faceapi.extractFaces(referenceImage, resultsRef);
    const faceImages2 = await faceapi.extractFaces(queryImage, resultsQuery);

    let distance = 1;

    if (faceImages1.length > 0 && faceImages2.length > 0) {
      const fim1 = await faceapi.computeFaceDescriptor(faceImages1[0]);
      const fim2 = await faceapi.computeFaceDescriptor(faceImages2[0]);

      distance = faceapi.utils.round(faceapi.euclideanDistance(fim1, fim2));
    } else {
      throw new Error('Sem rosto');
    }
    await fs.promises.unlink(originalPath);

    return {
      statusCode: 200,
      body: JSON.stringify({ distance, samePerson: distance < 0.5 }),
    };
  } catch (error) {
    const avatarFilename = req?.file?.filename;
    if (avatarFilename) {
      const originalPath = path.resolve(tmpFolder, avatarFilename);

      await fs.promises.unlink(originalPath);
    }
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message }),
    };
  }
};
