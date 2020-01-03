/* global Image */
import React, { useRef, useState, useEffect } from 'react'
import { render } from 'react-dom'

import pix2pix from './pix2pix'
import EdgeDetection from './EdgeDetection'

const Demo = () => {
  const refOut = useRef()
  const refEdge = useRef()
  const [ image, setImage ] = useState()
  const [ invert, setInvert ] = useState(false)
  const [ sigma, setSigma ] = useState(1.4)
  const [ size, setSize ] = useState(5)
  const [ lt, setLt ] = useState(50)
  const [ ht, setHt ] = useState(100)

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

  return (
    <div>
      <h2>pix2pix</h2>
      <p>Choose a square image file and this will detect the edges, then click an action to make a picture from that.</p>
      <div className='canvases'>
        <canvas ref={refEdge} height={256} width={256} />
        <div className='actions'>
          <button>cats!</button>
          <button>handbags!</button>
          <button>shoes!</button>
          <button>facade!</button>
        </div>
        <canvas ref={refOut} height={256} width={256} />
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
