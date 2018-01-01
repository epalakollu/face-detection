import socketio
import eventlet
import eventlet.wsgi
from flask import Flask, render_template

sio = socketio.Server()
app = Flask(__name__)


@app.route('/')
def index():
    """Serve the client-side application."""
    return "App running and working hard to emit events."

@sio.on('connect', namespace='/analytics')
def connect(sid, environ):
    print("connecteddd ", sid)


def sendMessage(sid,data):
    sio.emit('face time stats', data, namespace='/analytics')
    print('message sent',data)





@sio.on('statsdata')
def statsdata(sid,data):
    print("Message: ", data)
    sendMessage(sid,data)

@sio.on('send faces data')
def sendFacesData(sid,data):
    print("Faces Data: ", data)
    sio.emit('faces data', data, namespace='/analytics')    



@sio.on('disconnect', namespace='/analytics')
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 8000)), app)