/* global Image */
import React, { useRef, useState, useEffect } from 'react'
import { render } from 'react-dom'

import EdgeDetection from './EdgeDetection'
import { getWeights, applyWeights } from './pix2pix'

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
    const weights = await getWeights('/models/edges2cats_AtoB.pict')
    console.time('render')
    await applyWeights(weights, refEdge.current.getContext('2d'), refOut.current.getContext('2d'), SIZE)
    console.timeEnd('render')
  }

  const handleBags = async () => {
    const weights = await getWeights('/models/edges2handbags_AtoB.pict')
    console.time('render')
    await applyWeights(weights, refEdge.current.getContext('2d'), refOut.current.getContext('2d'), SIZE)
    console.timeEnd('render')
  }

  const handleShoes = async () => {
    const weights = await getWeights('/models/edges2shoes_AtoB.pict')
    console.time('render')
    await applyWeights(weights, refEdge.current.getContext('2d'), refOut.current.getContext('2d'), SIZE)
    console.timeEnd('render')
  }

  return (
    <div>
      <h2>pix2pix</h2>
      <p>Choose a square image file and this will detect the edges, then click an action to make a picture from that.</p>
      <p>You can save the canvases by right-clicking on them and choosing "Save Image As"</p>
      <div className='canvases'>
        <canvas ref={refEdge} height={SIZE} width={SIZE} />
        <div className='actions'>
          <button onClick={handleCats}>cats!</button>
          <button onClick={handleBags}>handbags!</button>
          <button onClick={handleShoes}>shoes!</button>
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
