import * as tf from '@tensorflow/tfjs';

export class ResizeLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.w = config.w;
        this.h = config.h;
      }
  
    call(input) {
      return tf.tidy(() => {
        return tf.image.resizeBilinear(input[0], [this.w, this.h]);
      });
      }
  
    computeOutputShape(input_shape) {
      return [input_shape[0], this.w, this.h, input_shape[3]]
    }
  
    static get className() {
      return 'ResizeLayer';
    }
}