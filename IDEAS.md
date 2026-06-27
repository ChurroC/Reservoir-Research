What if instead of the non linear springs being a reservoir basically could we derive the acoustic forces and use it as a way to back propogate.

What if instead of a single reservoir we connect it in almost layers like a neural network.
First I was thinking each layer takes the output and feeds it into the next reservoir.
But what about the gradient boosting method we take the residual or error and use that to feed into next layer.
We have each reservoir compensate for error loss from previously.
Compared to (raw data, layer n-1 output) I think the nth reservoir the error will force it to learn what's new or left not already what we know.