import { useRef, useEffect } from 'react'

const useCanvas = (draw) => {

  const canvasRef = useRef(null)

  useEffect(() => {

    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    let requestAnimationId
    let counter = 30
    const render = ctx => {
      ctx.save()
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      draw(ctx, counter)
      ctx.restore()
      //counter++
      requestAnimationId = requestAnimationFrame(() => render(ctx))
    }
    render(context)

    return () => {
      cancelAnimationFrame(requestAnimationId)
    }
  })

  return canvasRef
}

export default useCanvas
