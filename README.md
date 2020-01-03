# pix2pix-react

React hook to use [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) as a react component.

This is a more modern take on [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) that allows easier re-use. I'm using npm, ES6, react, and other modern web-tech, instead of globals, inlined vendor imports, and canvas-drawn UI.

## usage

It uses a submodule for the pre-trained models. You can get everything and start a demo webserver with these commands:

```
git clone --recurse git@github.com:konsumer/pix2pix-react.git
cd pix2pix-react
npm i
npm start
```