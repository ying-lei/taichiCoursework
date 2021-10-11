import taichi as ti

ti.init(ti.gpu)

# canvas
canvasSize = 512

# startup default param

searchRange = ti.field(ti.i32, shape=())
threshold = ti.field(ti.i32, shape=())
state = ti.field(ti.i32, shape=())

# range param
randRangeMin = 1
maxRange = 10
maxThreadhold = 25
maxState = 20

# startup value
searchRange[None] = 9
threshold[None] = 7
state[None] = 30

# color data storing image
pixels = ti.Vector.field(3, ti.f32, shape=(canvasSize, canvasSize))

# raw data storing states
raw = ti.field(dtype=int, shape=(canvasSize, canvasSize))


@ti.kernel
def initialize():

    for i, j in pixels:
        v = ti.random() * state[None]
        raw[i, j] = v

@ti.kernel
def randomParam():
    searchRange[None] = ti.random(ti.f32) * maxRange + randRangeMin
    threshold[None] = ti.random(ti.f32) * maxThreadhold + randRangeMin
    state[None] = ti.random(ti.f32) * maxState + randRangeMin



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
        remap = raw[i, j] / state[None]
        pixels[i, j] = (remap, remap, remap)



gui = ti.GUI('CCA2D', (canvasSize, canvasSize))

guiSearchRange = gui.slider('searchRange', randRangeMin, maxRange, step=1.0)
guiThreshold = gui.slider('threshold', randRangeMin, maxThreadhold, step=1.0)
guiState = gui.slider('state', randRangeMin, maxState, step=1.0)

redraw = gui.button('redraw')
randomize = gui.button('randomize')

def updateGUI():
    guiSearchRange.value = searchRange[None]
    guiThreshold.value = threshold[None]
    guiState.value = state[None]


updateGUI()
initialize()

while gui.running:

    for e in gui.get_events(gui.PRESS):
        if e.key == redraw:
            initialize()

        if e.key == randomize:
            randomParam()
            updateGUI()
            initialize()


    gui.clear(000000)
    update(int(guiSearchRange.value), int(guiThreshold.value), int(guiState.value))
    draw()
    gui.set_image(pixels)
    gui.show()