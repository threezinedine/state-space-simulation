from models.timer import Timer 
from models.signals import FullStateFeedBackController
from models.plants import Plant
from models import Model
import matplotlib.pyplot as plt
import numpy as np


def main():
    timer = Timer(start=0, stop=1, iterval=.01)
    time_arr = timer.get_time_input()
    initial_state = np.array([0.3, 0.2, 2.])
    plant = Plant(initial_state=initial_state)
    signals = FullStateFeedBackController(timer, feedback_model=plant, gain_param=np.array([1, 1, 1]))
    model = Model(plant)

    output = model.run(signals)
    print(output)

    for dimension in range(output.shape[1]):
        plt.plot(time_arr, output[:, dimension])

    plt.show()


if __name__ == "__main__":
    main()
