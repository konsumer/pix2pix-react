import React, { useRef } from 'react'
import { render } from 'react-dom'

import pix2pix from './pix2pix'

const Demo = () => {
  const refIn = useRef()
  const refOut = useRef()

  const handleFileChoose = (e) => {

  }

  return (
    <div>
      <div>
        <canvas ref={refIn} />
        <canvas ref={refOut} />
      </div>
      <div>
        <input type='file' onChange={handleFileChoose} />
      </div>
    </div>
  )
}

render(<Demo />, document.getElementById('root'))
