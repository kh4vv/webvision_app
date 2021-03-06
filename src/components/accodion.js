import React, {Component} from 'react';
import {Accordion, Button} from 'semantic-ui-react'
import {BrowserRouter as Router} from 'react-router-dom';


class AccordionStyled extends Component {
    state = { activeIndex: 0 }
  
    handleClick = (e, titleProps) => {
      const { index } = titleProps
      const { activeIndex } = this.state
      const newIndex = activeIndex === index ? -1 : index
  
      this.setState({ activeIndex: newIndex })
    }
  
    render() {
      const { activeIndex } = this.state;
  
      return (
        <Accordion>
        <Router>
          <Accordion.Title
            active={activeIndex === 0}
            index={0}
            onClick={this.handleClick}>
          <Button size="large" color="black"> Image Classification </Button>
          </Accordion.Title>
          <Accordion.Content active={activeIndex === 0}>
            <p>		
              <Button size="medium"><a href="/mnist"> MNIST </a></Button>
              </p><p>
              <Button size="medium"><a href="/quickdraw"> Quick Draw </a></Button>
              </p><p>
              <Button size="medium"><a href="/landmark">Korea Landmark </a></Button>
            </p>
          </Accordion.Content>
  
          <Accordion.Title
            active={activeIndex === 1}
            index={1}
            onClick={this.handleClick}>
            <Button size="large" color="black"> Object Detection </Button>
          </Accordion.Title>
          <Accordion.Content active={activeIndex === 1}>
            <p>
              <Button size="medium" color="grey"> One Stage </Button>
              <Button size="medium"> YOLO v1 </Button>
              </p><p>
              <Button size="medium"> YOLO v2 </Button>
              </p><p>
              <Button size="medium"><a href="/yolov3"> YOLO v3</a></Button>
              </p><p>
              <Button size="medium" color="grey"> Two Stage </Button>
              </p><p>
              <Button size="medium"> Faster RCNN </Button>
            </p>
          </Accordion.Content>
  
          <Accordion.Title
            active={activeIndex === 2}
            index={2}
            onClick={this.handleClick}>
          <Button size="large" color="black"> Instant Segmentation </Button>
          </Accordion.Title>
          <Accordion.Content active={activeIndex === 2}>
          <p>
          <Button size="medium"><a href="/maskrcnn"> Mask RCNN </a></Button>
          </p>
          </Accordion.Content>
          </Router>
        </Accordion>
      )
    }
  }

  export default AccordionStyled;