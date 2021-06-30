import * as tf from '@tensorflow/tfjs';
import { GPU } from 'gpu.js';
import * as cv from 'opencv.js';

import { ResizeLayer } from './layers';
import { recolor, rgbToHsv, hsvToRgb } from './recoloring';

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

        // this.gpu = new GPU();
        // this.recolor = this.gpu.createKernel(function(image, mask, texture) {
        //   const img = image[this.thread.y][this.thread.x];
        //   const tex = texture[this.thread.y][this.thread.x];

        //   if (mask[this.thread.y][this.thread.x] > 0.98) {
        //     // var r = img[0], g = img[1], b = img[2];


        //     // var max = Math.max(r, g, b), min = Math.min(r, g, b);
        //     // var v = max;

        //     const i_hsv = rgbToHsv(img[0], img[1], img[2]);
        //     const res_hsv = [tex[0], tex[1], i_hsv[2]];
        //     const res = hsvToRgb(res_hsv[0], res_hsv[1], res_hsv[2]);
        //     return res;

        //     // var h = tex[0];
        //     // var s = tex[1];
            
        //     // var i = Math.floor(h * 6);
        //     // var f = h * 6 - i;
        //     // var p = v * (1 - s);
        //     // var q = v * (1 - f * s);
        //     // var t = v * (1 - (1 - f) * s);
          
        //     // var mod = i % 6;
        //     // if (mod == 0) {
        //     //   r = v, g = t, b = p;
        //     // } else if (mod == 1) {
        //     //   r = q, g = v, b = p;
        //     // } else if (mod == 2) {
        //     //   r = p, g = v, b = t;
        //     // } else if (mod == 3) {
        //     //   r = p, g = q, b = v;
        //     // } else if (mod == 4) {
        //     //   r = t, g = p, b = v;
        //     // } else if (mod == 5) {
        //     //   r = v, g = p, b = q;
        //     // }

        //     // switch (i % 6) {
        //     //   case 0: r = v, g = t, b = p; break;
        //     //   case 1: r = q, g = v, b = p; break;
        //     //   case 2: r = p, g = v, b = t; break;
        //     //   case 3: r = p, g = q, b = v; break;
        //     //   case 4: r = t, g = p, b = v; break;
        //     //   case 5: r = v, g = p, b = q; break;
        //     // }
          
        //     // return [r, g, b];
        //   }
        //   return img;
        // }).setOutput([512, 512]).setFunctions([rgbToHsv, hsvToRgb]);
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
          const predArr = await tf.squeeze(predictions).array();

          // let imgmat = new cv.matFromArray(this.cameraSize[0], this.cameraSize[1], cv.CV_8UC3, imgArr);
          // console.log(imgmat);
          // console.log(imgmat.data);
          // return;
          const recolored = await recolor(imgArr, predArr, texture, this.cameraSize, false);

          // console.log(imgArr.isArray(), predArr.isArray(), texture.isArray());

          // const recolored = this.recolor(imgArr, predArr, texture);
          // console.log(imgArr);
          // console.log(recolored);
          // return -1;
        
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
          const predArr = await tf.squeeze(predictions).array();
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
          const predArr = await tf.squeeze(predictions).array();
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