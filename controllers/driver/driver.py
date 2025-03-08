"""
This controller gives to its node the following behavior:
Listen the keyboard. According to the pressed key, send a
message through an emitter or handle the position of Robot1.
"""

from controller import Supervisor
from common import common_print


class Driver (Supervisor):
    timeStep = 128
    x = -0.3
    y = -0.1
    translation = [x, y, 0]

    def __init__(self):
        super(Driver, self).__init__()
        self.emitter = self.getDevice('emitter')
        robot = self.getFromDef('ROBOT1')
        self.translationField = robot.getField('translation')
        self.keyboard.enable(Driver.timeStep)
        self.keyboard = self.getKeyboard()

    def run(self):
        self.displayHelp()
        previous_message = ''

        # Main loop.
        while True:
            # Deal with the pressed keyboard key.
            k = self.keyboard.getKey()
            message = ''
            if k == ord('A'):
                message = 'avoid obstacles'
            elif k == ord('F'):
                message = 'move forward'
            elif k == ord('S'):
                message = 'stop'
            elif k == ord('T'):
                message = 'turn'
            elif k == ord('I'):
                self.displayHelp()
            elif k == ord('G'):
                translationValues = self.translationField.getSFVec3f()
                print('ROBOT1 is located at (' + str(translationValues[0]) + ',' + str(translationValues[1]) + ')')
            elif k == ord('R'):
                print('Teleport ROBOT1 at (' + str(self.x) + ',' + str(self.y) + ')')
                self.translationField.setSFVec3f(self.translation)

            # Send a new message through the emitter device.
            if message != '' and message != previous_message:
                previous_message = message
                print('Please, ' + message)
                self.emitter.send(message.encode('utf-8'))

            # Perform a simulation step, quit the loop when
            # Webots is about to quit.
            if self.step(self.timeStep) == -1:
                break

    def displayHelp(self):
        print(
            'Commands:\n'
            ' I for displaying the commands\n'
            ' A for avoid obstacles\n'
            ' F for move forward\n'
            ' S for stop\n'
            ' T for turn\n'
            ' R for positioning ROBOT1 at (-0.3,-0.1)\n'
            ' G for knowing the (x,y) position of ROBOT1'
        )


controller = Driver()
common_print('driver')
controller.run()
