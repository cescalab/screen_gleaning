//https://github.com/JoeKarlsson/complete-guide-to-client-server-communication/tree/master/client-server-sockets
'use strict';

//https://stackoverflow.com/questions/16280747/sending-message-to-a-specific-connected-users-using-websocket/16320011
var webSocketServer = new (require('ws')).Server({port: (process.env.PORT || 8081)}),
    webSockets = {} // userID: webSocket

// CONNECT /:userID
// wscat -c ws://localhost:5000/1
webSocketServer.on('connection', function (webSocket) {
  var userID = parseInt(webSocket.upgradeReq.url.substr(1), 10)
  webSockets[userID] = webSocket
  console.log('connected: ' + userID + ' in ' + Object.getOwnPropertyNames(webSockets))

  // Forward Message
  //
  // Receive               Example
  // [toUserID, text]      [2, "Hello, World!"]
  //
  // Send                  Example
  // [fromUserID, text]    [1, "Hello, World!"]
  webSocket.on('message', function(message) {
    console.log('received from ' + userID + ': ' + message)
    var messageArray = JSON.parse(message)
    var toUserWebSocket = webSockets[messageArray[0]]
    if (toUserWebSocket) {
      console.log('sent to ' + messageArray[0] + ': ' + JSON.stringify(messageArray))
      messageArray[0] = userID
      toUserWebSocket.send(JSON.stringify(messageArray))
    }
  })

  webSocket.on('close', function () {
    delete webSockets[userID]
    console.log('deleted: ' + userID)
  })
})
