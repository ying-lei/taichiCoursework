import taichi as ti

ti.init(ti.gpu)

# canvas
canvasSize = 512

# startup default param
searchRange = 9
threshold = 7
state = 30

#pixels = ti.Vector.field(3, ti.f32, shape=(canvasSize, canvasSize))

# color data storing image
pixels = ti.Vector.field(3, ti.f32, shape=(canvasSize, canvasSize))

# raw data storing states
raw = ti.field(dtype=int, shape=(canvasSize, canvasSize))


@ti.kernel
def initialize():

    for i, j in pixels:
        v = ti.random() * state
        raw[i, j] = v

@ti.kernel
def update(searchRange:int, threshold:int, state:int):
    for i, j in raw:
        currentState = raw[i, j] 
        nextState = 0 if (currentState + 1 == state) else (currentState + 1)
        count = 0

        for x in range(-searchRange, searchRange):
            for y in range(-searchRange, searchRange):

                # ignore self
                if (x==0 and y==0):
                    pass

                elif(raw[i+x, j+y] == nextState):
                    count += 1

                else:
                    pass

        if count >= threshold:
            raw[i,j] = nextState



@ti.kernel
def draw():
    # color remap code here
    for i, j in pixels:
        remap = raw[i, j] / state
        pixels[i, j] = (remap, remap, remap)



gui = ti.GUI('CCA2D', (canvasSize, canvasSize))

guiSearchRange = gui.slider('searchRange', 1, 50, step=1)
guiSearchRange.value = searchRange

guiThreshold = gui.slider('threshold', 1, 50, step=1)
guiThreshold.value = threshold

guiState = gui.slider('state', 1, 50, step=1)
guiState.value = state

redraw = gui.button('redraw')

initialize()

while gui.running:

    for e in gui.get_events(gui.PRESS):
        if e.key == redraw:
            initialize()

    gui.clear(000000)
    update(int(guiSearchRange.value), int(guiThreshold.value), int(guiState.value))
    draw()
    gui.set_image(pixels)
    gui.show()