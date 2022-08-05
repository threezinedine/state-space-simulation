from cores import IEvent
from views import Window
import unittest
from unittest.mock import Mock
import pytest
import pygame



class TestUserInterface(unittest.TestCase):
    def test_exist_the_window(self):
        win = Window()
        event = Mock(spec=IEvent)
        event.get.return_value = [pygame.QUIT]

        while win.is_still_running():
            event.detect()
            win.handle_event(event)
            win.draw()

        self.assertFalse(win.is_still_running())
