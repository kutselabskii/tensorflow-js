export function rgbToHsv(r, g, b) {
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

function getV(img) {
  return Math.max(img[0], img[1], img[2])
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
  
export async function recolor(image, mask, texture, size, inverted) {
    for (let i = 0; i < size[0]; ++i) {
      const tex_a = texture[i];
      const img_a = image[i];
      for (let j = 0; j < size[1]; ++j) {
        const tex = tex_a[j];
        
        if (!inverted && mask[i][j] > 0.98 || inverted && mask[i][j] < 0.3) {
          const res_hsv = [tex[0], tex[1], getV(img_a[j])];
          image[i][j] = hsvToRgb(res_hsv[0], res_hsv[1], res_hsv[2]);
        }
      }
    }
    return image;
}