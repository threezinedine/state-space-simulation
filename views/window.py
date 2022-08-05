import pygame


SCREEN_SIZE = (400, 300)
CAPTION = "Simulation"
QUIT = pygame.QUIT



class Window:
    def __init__(self, screen_size=SCREEN_SIZE, caption=CAPTION):
        self._running = True
        self.screen = None
        self._size = screen_size
        self._caption = caption
        self._setup()

    def _setup(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self._size)
        pygame.display.set_caption(self._caption)

    def is_still_running(self) -> bool:
        return self._running

    def handle_event(self, event) -> None:
        for e in event.get():
            if e == QUIT:
                self._running = False

    def draw(self) -> None:
        pygame.display.update()
