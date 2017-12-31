import socketio
import eventlet
import eventlet.wsgi
from flask import Flask, render_template

sio = socketio.Server()
app = Flask(__name__)


@app.route('/')
def index():
    """Serve the client-side application."""
    return "Hi, EK"

@sio.on('connect', namespace='/analytics')
def connect(sid, environ):
    print("connecteddd ", sid)


def sendMessage(sid,data):
    sio.emit('my response', data, namespace='/analytics')

    print('message sent',data)


@sio.on('statsdata')
def statsdata(sid,data):
    print("Message: ", data)
    #print("message ", data)
    sendMessage(sid,data)

@sio.on('disconnect', namespace='/analytics')
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 8000)), app)