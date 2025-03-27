# BSLAPI Server

This folder is a webserver meant to wrap around the BSLAPI tools provided in the `bslapi` package.
The aim is to provide a web client interface for these services, so that websites can interact with it more easily.

The server will have the following features:

- Map all methods provided by the inner API.
- Be responsible for decoding and encoding requests and responses for web clients.
- Handle the JSON from client to server to api and back.
- Websocket connection for streaming.

## Methods

- Recognize Image
- Recognize Video Stream
- WS Recognize Image
- WS Reconigze Video

