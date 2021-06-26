# Introduction to Lévyprocesses and financialapplications
=========================================================

This repository complements my master's thesis on introduction to Lévy processes with applications in finance. 

It serves as a guide to see how to implement almost from scratch, model selection and valuation of European options using Lévy exponential models. Lewis Fourier transform method and Monte Carlo are used.

The code is free of usage, but the author does not take responsibility for its use in practice since there may be some minor bugs.

## Contents

<ul>
<li>utils.py : Lewis Fourier transform method and functions for the payoffs</li>
<li>mertonPricer: Class for model fitting and option valuation using Merton’s model</li>
<li>alPhaStablePricer.py : Class for model fitting and option valuation using alpha stable model</li>
<li>ghPricer.py : Class for model fitting and option valuation using generalized hyperbolic model (It contains some bugs. Although the fit part seems to be fine, in option valuation unexpected results are seen!)</li>
<li>TFM_INAKI.ipynb : Notebook with the graphs and some results shown in the work</li>
<li>Tests.ipynb : 
Notebooks with test for Merton and alpha stable
</li>
</ul>

## References

[1] R Cont and P Tankov.Financial Modelling with Jump Processes. Chapman & Hall/CRC,2003.

[2] Alan L. Lewis.   A Simple Option Formula for General Jump-Diusion and otherExponential Levy Processes. Related articles explevy, Finance Press, August 2001.

## Contributing
 
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :+1:
 
## Credits
 
Author: Iñaki Gorostiaga (innaki.gorostiaga@estudiante.uam.es)
 
## License
 
The MIT License (MIT)

Copyright (c) 2021 Iñaki Gorostiaga

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.