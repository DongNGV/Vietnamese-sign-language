import pyttsx3
from utils.tao_cau import *

def SpeakText(inputText):
    engine = pyttsx3.init()
    voice = engine.getProperty("voices")
    engine.setProperty("voice", voice[1].id)
    engine.setProperty("rate", 150)

    inputText = TaoCauHoanChinh(inputText)

    engine.say(inputText)
    engine.runAndWait()

