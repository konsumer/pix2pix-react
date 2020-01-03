/* global Image, fetch, TextDecoder, ImageData */
import React, { useRef, useState, useEffect } from 'react'
import { render } from 'react-dom'
import * as dl from 'deeplearn'

import EdgeDetection from './EdgeDetection'

const getWeights = async (model) => {
  const r = await fetch(`/models/${model}.pict`, { cache: 'force-cache' })
  const buf = await r.arrayBuffer()
  const parts = []
  let offset = 0
  while (offset < buf.byteLength) {
    var b = new Uint8Array(buf.slice(offset, offset + 4))
    offset += 4
    var len = (b[0] << 24) + (b[1] << 16) + (b[2] << 8) + b[3]
    parts.push(buf.slice(offset, offset + len))
    offset += len
  }
  var shapes = JSON.parse((new TextDecoder('utf8')).decode(parts[0]))
  var index = new Float32Array(parts[1])
  var encoded = new Uint8Array(parts[2])
  // decode using index
  var arr = new Float32Array(encoded.length)
  for (var i = 0; i < arr.length; i++) {
    arr[i] = index[encoded[i]]
  }
  var weights = {}
  offset = 0
  for (let i = 0; i < shapes.length; i++) {
    var shape = shapes[i].shape
    var size = shape.reduce((total, num) => total * num)
    var values = arr.slice(offset, offset + size)
    var dlarr = dl.Array1D.new(values, 'float32')
    weights[shapes[i].name] = dlarr.reshape(shape)
    offset += size
  }
  return weights
}

function model (input, weights) {
  const math = dl.ENV.math
  function preprocess (input) {
    return math.subtract(math.multiply(input, dl.Scalar.new(2)), dl.Scalar.new(1))
  }
  function deprocess (input) {
    return math.divide(math.add(input, dl.Scalar.new(1)), dl.Scalar.new(2))
  }
  function batchnorm (input, scale, offset) {
    var moments = math.moments(input, [0, 1])
    const varianceEpsilon = 1e-5
    return math.batchNormalization3D(input, moments.mean, moments.variance, varianceEpsilon, scale, offset)
  }
  function conv2d (input, filter, bias) {
    return math.conv2d(input, filter, bias, [2, 2], 'same')
  }
  function deconv2d (input, filter, bias) {
    var convolved = math.conv2dTranspose(input, filter, [input.shape[0] * 2, input.shape[1] * 2, filter.shape[2]], [2, 2], 'same')
    var biased = math.add(convolved, bias)
    return biased
  }
  var preprocessed_input = preprocess(input)
  var layers = []
  var filter = weights['generator/encoder_1/conv2d/kernel']
  var bias = weights['generator/encoder_1/conv2d/bias']
  var convolved = conv2d(preprocessed_input, filter, bias)
  layers.push(convolved)
  for (let i = 2; i <= 8; i++) {
    var scope = 'generator/encoder_' + i.toString()
    var filter = weights[scope + '/conv2d/kernel']
    var bias = weights[scope + '/conv2d/bias']
    var layer_input = layers[layers.length - 1]
    var rectified = math.leakyRelu(layer_input, 0.2)
    var convolved = conv2d(rectified, filter, bias)
    var scale = weights[scope + '/batch_normalization/gamma']
    var offset = weights[scope + '/batch_normalization/beta']
    var normalized = batchnorm(convolved, scale, offset)
    layers.push(normalized)
  }
  for (let i = 8; i >= 2; i--) {
    if (i === 8) {
      var layer_input = layers[layers.length - 1]
    } else {
      var skip_layer = i - 1
      var layer_input = math.concat3D(layers[layers.length - 1], layers[skip_layer], 2)
    }
    var rectified = math.relu(layer_input)
    var scope = 'generator/decoder_' + i.toString()
    var filter = weights[scope + '/conv2d_transpose/kernel']
    var bias = weights[scope + '/conv2d_transpose/bias']
    var convolved = deconv2d(rectified, filter, bias)
    var scale = weights[scope + '/batch_normalization/gamma']
    var offset = weights[scope + '/batch_normalization/beta']
    var normalized = batchnorm(convolved, scale, offset)
    // missing dropout
    layers.push(normalized)
  }
  var layer_input = math.concat3D(layers[layers.length - 1], layers[0], 2)
  var rectified = math.relu(layer_input)
  var filter = weights['generator/decoder_1/conv2d_transpose/kernel']
  var bias = weights['generator/decoder_1/conv2d_transpose/bias']
  var convolved = deconv2d(rectified, filter, bias)
  var rectified = math.tanh(convolved)
  layers.push(rectified)
  var output = layers[layers.length - 1]
  var deprocessed_output = deprocess(output)
  return deprocessed_output
}

const Demo = () => {
  const refOut = useRef()
  const refEdge = useRef()
  const [ image, setImage ] = useState()
  const [ invert, setInvert ] = useState(false)
  const [ sigma, setSigma ] = useState(1.4)
  const [ size, setSize ] = useState(5)
  const [ lt, setLt ] = useState(50)
  const [ ht, setHt ] = useState(100)

  const SIZE = 256

  useEffect(() => {
    if (image) {
      const edge = new EdgeDetection(refEdge.current)
      edge.loadImage(image.src)
      setTimeout(() => {
        // edge-detection
        edge.resetImage()
        edge.greyscale()
        edge.gaussian(sigma, size)
        edge.sobel()
        edge.drawOnCanvas()
        edge.nonMaximumSuppression()
        edge.hysteresis(lt, ht)
        if (invert) {
          edge.invert()
        }
        //
      }, 0)
    }
  }, [image, invert, sigma, size, lt, ht])

  const handleFileChoose = async (eFile) => {
    const i = new Image()
    i.src = URL.createObjectURL(eFile.target.files[0])
    i.onload = () => setImage(i)
  }

  const handleInvert = () => setInvert(i => !i)
  const handleSigma = e => setSigma(e.target.value)
  const handleSize = e => setSize(e.target.value)
  const handleLt = e => setLt(e.target.value)
  const handleHt = e => setHt(e.target.value)

  const handleCats = async () => {
    const weights = await getWeights('edges2cats_AtoB')
    var input_uint8_data = refEdge.current.getContext('2d').getImageData(0, 0, SIZE, SIZE).data
    var input_float32_data = Float32Array.from(input_uint8_data, (x) => x / 255)
    console.time('render')
    const math = dl.ENV.math
    math.startScope()
    var input_rgba = dl.Array3D.new([SIZE, SIZE, 4], input_float32_data, 'float32')
    var input_rgb = math.slice3D(input_rgba, [0, 0, 0], [SIZE, SIZE, 3])
    var output_rgb = model(input_rgb, weights)
    var alpha = dl.Array3D.ones([SIZE, SIZE, 1])
    var output_rgba = math.concat3D(output_rgb, alpha, 2)
    output_rgba.getValuesAsync().then((output_float32_data) => {
      var output_uint8_data = Uint8ClampedArray.from(output_float32_data, (x) => x * 255)
      refOut.current.getContext('2d').putImageData(new ImageData(output_uint8_data, SIZE, SIZE), 0, 0)
      math.endScope()
      console.timeEnd('render')
    })
  }

  return (
    <div>
      <h2>pix2pix</h2>
      <p>Choose a square image file and this will detect the edges, then click an action to make a picture from that.</p>
      <div className='canvases'>
        <canvas ref={refEdge} height={SIZE} width={SIZE} />
        <div className='actions'>
          <button onClick={handleCats}>cats!</button>
          <button>handbags!</button>
          <button>shoes!</button>
          <button>facade!</button>
        </div>
        <canvas ref={refOut} height={SIZE} width={SIZE} />
      </div>
      <div className='controls'>
        <label>
          Image File: <input type='file' onChange={handleFileChoose} />
        </label>
        <label>
          Invert? <input type='checkbox' checked={invert} onChange={handleInvert} />
        </label>
        <label>
          Sigma <input min={0} max={10} step={0.1} type='range' value={sigma} onChange={handleSigma} /> ({sigma})
        </label>
        <label>
          Size <input min={0} max={20} step={0.1} type='range' value={size} onChange={handleSize} /> ({size})
        </label>
        <label>
          LT <input min={0} max={200} type='range' value={lt} onChange={handleLt} /> ({lt})
        </label>
        <label>
          HT <input min={0} max={200} type='range' value={ht} onChange={handleHt} /> ({ht})
        </label>
      </div>
    </div>
  )
}

render(<Demo />, document.getElementById('root'))
