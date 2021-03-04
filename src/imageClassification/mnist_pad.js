import React, { Component } from 'react'
import SignaturePad from 'react-signature-canvas'
import { Button, Header, Icon } from 'semantic-ui-react';
import axios from "axios";

import styles from './styles.module.css'

class Pad extends Component {
  state = { value: null, predic: null }
  sigPad = {}
  clear = () => {
    this.sigPad.clear()
  }

  fileUploadHandler = async event => {
    event.preventDefault();
    this.setState({
      value: this.sigPad.getCanvas()
        .toDataURL('image/png')
    })
    const fd = new FormData()
    fd.append('url', this.state.value)
    axios
      .post("http://localhost:9000/mnist_pad", fd, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((res) => {
        this.setState({ predic: res.data.pred });
        console.log(this.state.predic)
      });
  };

  render() {
    let { value } = this.state
    return <div className={styles.container}>
      <h2><Icon name="edit" /> Draw Yourself! </h2>
      <div className={styles.sigContainer}>
        <SignaturePad maxWidth="50" canvasProps={{ className: styles.sigPad }}
          ref={(ref) => { this.sigPad = ref }} />
      </div>
      <div>
        <Button.Group>
          <Button onClick={this.clear}>  Clear    </Button>
          <Button.Or />
          <Button positive onClick={e => this.fileUploadHandler(e)}> Submit </Button></Button.Group>
      </div>
      <Header as="h2">
        <Icon name="plug" />
        <Header.Content> Prediction : {this.state.predic}</Header.Content>
      </Header>
    </div>
  }
}

export default Pad;

