import { onMounted, onUnmounted } from 'vue'

function canScrollHorizontally(target: EventTarget | null, deltaX: number): boolean {
  let el = target instanceof Element ? target : null

  while (el && el !== document.documentElement && el !== document.body) {
    const style = window.getComputedStyle(el)
    const overflowX = style.overflowX
    const canScroll = /(auto|scroll|overlay)/.test(overflowX) && el.scrollWidth > el.clientWidth

    if (canScroll) {
      const maxScrollLeft = el.scrollWidth - el.clientWidth
      if (deltaX < 0 && el.scrollLeft > 0) return true
      if (deltaX > 0 && el.scrollLeft < maxScrollLeft) return true
    }

    el = el.parentElement
  }

  return false
}

export function usePreventNavigationSwipe() {
  function onWheel(event: WheelEvent) {
    if (event.ctrlKey) return

    const horizontal = Math.abs(event.deltaX)
    const vertical = Math.abs(event.deltaY)
    const isHorizontalIntent = horizontal > 8 && horizontal > vertical * 1.15

    if (!isHorizontalIntent || canScrollHorizontally(event.target, event.deltaX)) return

    if (event.cancelable) event.preventDefault()
  }

  onMounted(() => {
    window.addEventListener('wheel', onWheel, { passive: false, capture: true })
  })

  onUnmounted(() => {
    window.removeEventListener('wheel', onWheel, { capture: true })
  })
}
