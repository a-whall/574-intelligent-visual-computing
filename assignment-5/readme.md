The best average reconstructed test error I could get was 0.029 and the images in /results
are the output from that model for visually inspecting the coherency of the output.

I tried using the model output to update a single pixel at each iteration of the reconstruction
loop in a row-wise auto-regressive manner and got errors in the range of [0.031, 0.078].

I then tried using the model output to update the entire distorted region up to the current
pixel at each iteration of the loop and got errors in a similar range so I couldn't tell if one
is better than the other. The best test error came from this method, though.