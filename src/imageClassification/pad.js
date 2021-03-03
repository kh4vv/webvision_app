import React, { Component } from 'react'
import SignaturePad from 'react-signature-canvas'
import { Button } from 'semantic-ui-react';
import axios from "axios";

import styles from './styles.module.css'

class Pad extends Component {
  state = { value: null }
  sigPad = {}
  clear = () => {
    this.sigPad.clear()
  }

  fileUploadHandler = () => {
    this.setState({
      value: this.sigPad.getCanvas()
        .toDataURL('image/png')
    })
    console.log(this.state.value);
    const fd = new FormData()
    fd.append('url', this.state.value)
    axios
      .post("http://localhost:9000/mnist_pad", fd, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((res) => {
        console.log(res);
      });
  };

  render() {
    let { value } = this.state
    return <div className={styles.container}>
      <div className={styles.sigContainer}>
        <SignaturePad maxWidth="50" canvasProps={{ className: styles.sigPad }}
          ref={(ref) => { this.sigPad = ref }} />
      </div>
      <div>
        <Button.Group>
          <Button onClick={this.clear}>  Clear    </Button>
          <Button.Or />
          <Button positive onClick={this.fileUploadHandler}> Submit </Button></Button.Group>
      </div>
      {value
        ? <img className={styles.sigImage}
          src={value} alt="" />
        : null}

    </div>
  }
}

export default Pad;
//ReactDOM.render(<Pad />, document.getElementById('root'))

