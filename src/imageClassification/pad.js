import React, { Component } from 'react'
import SignaturePad from 'react-signature-canvas'
import {Button} from 'semantic-ui-react';

import styles from './styles.module.css'

class Pad extends Component {
  state = {trimmedDataURL: null}
  sigPad = {}
  clear = () => {
    this.sigPad.clear()
  }
  trim = () => {
    this.setState({trimmedDataURL: this.sigPad.getTrimmedCanvas()
      .toDataURL('image/png')})
  }
  render () {
    let {trimmedDataURL} = this.state
    return <div className={styles.container}>
      <div className={styles.sigContainer}>
        <SignaturePad canvasProps={{className: styles.sigPad}}
          ref={(ref) => { this.sigPad = ref }} />
      </div>
      <div>
	<Button.Group>
        <Button onClick={this.clear}>  Clear    </Button>
	<Button.Or />  
        <Button positive onClick={this.trim}> Submit </Button></Button.Group>
      </div>
      {trimmedDataURL
        ? <img className={styles.sigImage}
          src={trimmedDataURL} />
        : null}
      
    </div>
  }
}

export default Pad;
//ReactDOM.render(<Pad />, document.getElementById('root'))

