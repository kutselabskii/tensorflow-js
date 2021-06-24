export function rgbToHsv(r, g, b) {
    r /= 255, g /= 255, b /= 255;
  
    var max = Math.max(r, g, b), min = Math.min(r, g, b);
    var h, s, v = max;
  
    var d = max - min;
    s = max == 0 ? 0 : d / max;
  
    if (max == min) {
      h = 0; // achromatic
    } else {
      switch (max) {
        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
        case g: h = (b - r) / d + 2; break;
        case b: h = (r - g) / d + 4; break;
      }
  
      h /= 6;
    }
  
    return [ h, s, v ];
}
  
function hsvToRgb(h, s, v) {
    var r, g, b;
  
    var i = Math.floor(h * 6);
    var f = h * 6 - i;
    var p = v * (1 - s);
    var q = v * (1 - f * s);
    var t = v * (1 - (1 - f) * s);
  
    switch (i % 6) {
      case 0: r = v, g = t, b = p; break;
      case 1: r = q, g = v, b = p; break;
      case 2: r = p, g = v, b = t; break;
      case 3: r = p, g = q, b = v; break;
      case 4: r = t, g = p, b = v; break;
      case 5: r = v, g = p, b = q; break;
    }
  
    return [r, g, b];
}
  
export async function recolor(image, mask, texture, size) {
    for (let i = 0; i < size[0]; ++i) {
      for (let j = 0; j < size[1]; ++j) {
        const img = image[i][j];
        
        if (mask[0][i][j][0] > 0.98) {
          const i_hsv = rgbToHsv(img[0], img[1], img[2]);
          const res_hsv = [texture[i][j][0], texture[i][j][1], i_hsv[2]];
          image[i][j] = hsvToRgb(res_hsv[0], res_hsv[1], res_hsv[2]);
        } else {
          image[i][j] = [img[0] / 255, img[1] / 255, img[2] / 255];
        }
      }
    }
    return image;
}