Experimental setup for collecting emages.

1. Choose a front-end (your device).
2. Choose an antenna.
3. Put the antenna on the FPC (flexible circuit board) part of phone.
4. Open the GUI, go to File and choose Load USRP.
5. Enter the command "–rate=" to set sampling rate or use the default setting.
6. Press start button.

Searching process.

7. Increase the sampling rate every time one unit.
8. Press "AUT" button to automatically detect the resolution and line rate.
9. Adjust Framerate or Video height and check the results with eyes in monitor.
10. Repeat step 7. to 9. until the emage is visible on monitor.

----------------------------------------------------------------------------------------------------
Correct values found for Iphone 6S:
- frequency: 300 MHz.
- probe position: top-right corner of the phone, aligned with the selfie camera (position of the junction of the circle and the handle of the probe. The circle turned to the middle-bottom of the phone.)

CAUTION. The framerate and the video height detection values should be carefully selected (see screen specification for width and height). 

----------------------------------------------------------------------------------------------------

Running automated experiment with javascript server:
1. Run http server with the command : http-server
2. Run the node.js server : node server.js
3. Connect the phone to the local network via WIFI.
3. Open the webpage at the localhost URL of the http server (e.g., http://192.168.1.2) (Make sure to refresh the page after starting the node server, otherwise, the device will not be connected.)
if the localhost URL is different (probably because the router restarted?) check IP with ifconfig and you might also need to change website URL in file /var/www/emage/html/images/index.html
4. Switch the view of the browser in full screen with icon on the bottom right corner and adapt image to fit the screen.
5. Run the TempestSDR application and tune the parameters as explained above.
6. Go to "Tweaks -> connect to server", and the measurement acquiston will start.

