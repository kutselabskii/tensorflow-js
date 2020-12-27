const tf = require('@tensorflow/tfjs');
const http = require('http')
const fs = require('fs')

require('@tensorflow/tfjs-node');

// const model = tf.sequential();
// model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}));
// model.add(tf.layers.dense({units: 1, activation: 'linear'}));
// model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// const xs = tf.randomNormal([100, 10]);
// const ys = tf.randomNormal([100, 1]);

// model.fit(xs, ys, {
//   epochs: 100,
//   callbacks: {
//     onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
//   }
// });

const server = http.createServer((req, res) => {
  res.writeHead(200)
  fs.createReadStream('index.html').pipe(res)
})

server.listen(process.env.PORT || 3000)