import eel
eel.init('web')

@eel.expose
def add():
    return 1

eel.start('dashboard.html')