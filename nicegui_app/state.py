import requests
import json
import re

class AppState:
    def __init__(self):
        self.specialization = 'statistics'
        self.engine_mode = 'single'
        self.messages = []
        self.pipeline_steps = []
        self.is_processing = False

    def clear_chat(self):
        self.messages = []
        self.pipeline_steps = []

    def set_specialization(self, spec: str):
        self.specialization = spec
        self.clear_chat()

    def set_engine_mode(self, mode: bool):
        self.engine_mode = 'multi' if mode else 'single'

state = AppState()
