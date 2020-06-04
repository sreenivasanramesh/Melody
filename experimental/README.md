

A different approach using a different pre-processing/encoding technique. 
Instead of using chords to encode the music, we encode it as a sequence of key presses, 
similar to the work done by [Christine McLeavey](http://christinemcleavey.com/clara-a-neural-net-music-generator/).

The main advantage of using this is the reduced number of classes. We also don't need to handle "unseen" data in the 
prompt given for generation.

However, this is not providing better results currently. Have to figure out why.

To run this, just copy these files to the main directory.