import * as tf from '@tensorflow/tfjs';
import { ResizeLayer } from './layers';
import { recolor } from './recoloring';

export class LinkNet {
    constructor(cameraSize) {
        this.cameraSize = cameraSize;

        this.size = [128, 128];
        this.name = "LinkNet";
        this.model = undefined;
        this.webcam = undefined;
        this.canvas = undefined;
    }

    async load() {
        this.model = await tf.loadLayersModel("./" + this.name + "/model.json");
    }

    setScreens(webcam, canvas) {
        this.webcam = webcam;
        this.canvas = canvas;
    }

    async predict(texture) {
        if (this.webcam === undefined || texture === undefined) {
            return -1;
          }
        
          const start = performance.now();
        
          const original = await this.webcam.capture();
          const img = tf.image.resizeBilinear(original, this.cameraSize);
        
          const predictions = tf.tidy(() => {
            const preds = tf.image.resizeBilinear(img, this.size).reshape([1, this.size[0], this.size[1], 3]);
            return tf.image.resizeBilinear(this.model.predict(preds), this.cameraSize);
        
            // const preds = tf.add(tf.mul(tf.cast(img, 'float32'), 2 / 255), -1);
            // const preds = tf.div(tf.cast(img, 'float32'), 255);
          });
        
          const imgArr = await img.div(255).array();
          const predArr = await predictions.array();
          const recolored = await recolor(imgArr, predArr, texture, this.cameraSize, false);
        
          tf.tidy(() => {
            tf.browser.toPixels(tf.tensor(recolored), this.canvas);
          });
        
          original.dispose();
          img.dispose();
          predictions.dispose();

          return performance.now() - start;
    }
}

export class FastSCNN {
    constructor(cameraSize) {
        this.cameraSize = cameraSize;

        this.size = [512, 512];
        this.name = "Fast-SCNN";
        this.model = undefined;
        this.webcam = undefined;
        this.canvas = undefined;
    }

    async load() {
        tf.serialization.registerClass(ResizeLayer);
        this.model = await tf.loadLayersModel("./" + this.name + "/model.json");
    }

    setScreens(webcam, canvas) {
        this.webcam = webcam;
        this.canvas = canvas;
    }

    async predict(texture) {
        if (this.webcam === undefined || texture === undefined) {
            return -1;
          }
        
          const start = performance.now();
        
          const original = await this.webcam.capture();
          const img = tf.image.resizeBilinear(original, this.cameraSize).div(255);
        
          const predictions = tf.tidy(() => {
            const preds = tf.image.resizeBilinear(img, this.size).reshape([1, this.size[0], this.size[1], 3]);
            return tf.image.resizeBilinear(this.model.predict(preds), this.cameraSize);
          });
        
          const imgArr = await img.array();
          const predArr = await predictions.array();
          const recolored = await recolor(imgArr, predArr, texture, this.cameraSize, true);
        
          tf.tidy(() => {
            tf.browser.toPixels(tf.tensor(recolored), this.canvas);
          });
        
          original.dispose();
          img.dispose();
          predictions.dispose();

          return performance.now() - start;
    }
}

export class UNet {
    constructor(cameraSize) {
        this.cameraSize = cameraSize;

        this.size = [128, 128];
        this.name = "U-Net";
        this.model = undefined;
        this.webcam = undefined;
        this.canvas = undefined;
    }

    async load() {
        tf.serialization.registerClass(ResizeLayer);
        this.model = await tf.loadLayersModel("./" + this.name + "/model.json");
    }

    setScreens(webcam, canvas) {
        this.webcam = webcam;
        this.canvas = canvas;
    }

    async predict(texture) {
        if (this.webcam === undefined || texture === undefined) {
            return -1;
          }
        
          const start = performance.now();
        
          const original = await this.webcam.capture();
          const img = tf.image.resizeBilinear(original, this.cameraSize).div(255);
        
          const predictions = tf.tidy(() => {
            const preds = tf.image.resizeBilinear(img, this.size).reshape([1, this.size[0], this.size[1], 3]);
            return tf.image.resizeBilinear(this.model.predict(preds), this.cameraSize);
          });
        
          const imgArr = await img.array();
          const predArr = await predictions.array();
          const recolored = await recolor(imgArr, predArr, texture, this.cameraSize, true);
        
          tf.tidy(() => {
            tf.browser.toPixels(tf.tensor(recolored), this.canvas);
          });
        
          original.dispose();
          img.dispose();
          predictions.dispose();

          return performance.now() - start;
    }
}