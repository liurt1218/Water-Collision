CC          = g++ 

#-----------------------------------------
#Optimization ----------------------------
OPT   =  -O3

#-----------------------------------------

TARGETS = Array3DTest StaggeredGridTest FluidSimulator ParticleViewer

OBJECTS = jsoncpp.o Particle.o PressureSolver.o SimulationParameters.o StaggeredGrid.o

#-----------------------------------------

CCOPTS = $(OPT) -I/usr/include/eigen3/
LDOPTS = 

#-----------------------------------------
#-----------------------------------------

default: $(TARGETS)

clean:
	/bin/rm -f *.o $(TARGETS)

#-----------------------------------------
#-----------------------------------------

Array3DTest: Array3DTest.o
	$(CC) Array3DTest.o -o Array3DTest

StaggeredGridTest: $(OBJECTS) StaggeredGridTest.o
	$(CC) $(OBJECTS) StaggeredGridTest.o -o StaggeredGridTest

FluidSimulator: $(OBJECTS) FluidSimulator.o
	$(CC) $(OBJECTS) $(LDOPTS) FluidSimulator.o -o FluidSimulator

ParticleViewer: ParticleViewer.o
	$(CC) ParticleViewer.o -o ParticleViewer -L/usr/lib/x86_64-linux-gnu/ -lGL -lglut -lGLU

#-----------------------------------------
#-----------------------------------------

.cpp.o: 
	$(CC) $(CCOPTS) -c $< 

#-----------------------------------------
#-----------------------------------------


