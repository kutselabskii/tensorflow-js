import * as tf from '@tensorflow/tfjs';

import cameraIcon from '../img/camera.png';
import sofaIcon from '../img/sofa.png';
import empty from '../img/chess.png';
import t1 from '../img/1.jpg';
import t2 from '../img/2.jpg';
import t3 from '../img/3.jpg';
import e1 from '../img/e1.jpg';
import e2 from '../img/e2.jpg';
import e3 from '../img/e3.jpg';
import e4 from '../img/e4.jpg';

import { rgbToHsv } from './recoloring';


export const textures = [t1, t2, t3];
export const examples = [e1, e2, e3, e4];
export var tensors = [];

export function prepareTextureTensors(size, htmlImages) {
    textures.forEach(texture => {
        const image = new Image();
        const enclosedTexture = texture;
        image.crossOrigin = 'anonymous';

        image.onload = () => {
            const tensor = tf.tidy(() => {
                var imageTensor = tf.browser.fromPixels(image);
                imageTensor = tf.image.resizeBilinear(imageTensor, size).arraySync();
                for (let i = 0; i < size[0]; ++i) {
                    for (let j = 0; j < size[1]; ++j) {
                        imageTensor[i][j] = rgbToHsv(imageTensor[i][j][0], imageTensor[i][j][1], imageTensor[i][j][2]);
                    }
                }
                return imageTensor;
            });

            tensors.push(tensor);
            htmlImages[tensors.length - 1].src = enclosedTexture;
        };

        image.src = texture;
    });
}