import React, { Component } from 'react';
import { Header, Icon } from "semantic-ui-react";

class Welcome extends Component {

    render() {
        return (
            <div>
                <br></br>
                <Header as='h1' icon textAlign='center'>
                    <Icon name='computer' circular />
                    <Header.Content>Weclome to Computer Vision World</Header.Content>
                </Header>
                <p style={{ textAlign: "center", fontSize: "20px" }}>
                    Hello! This is K.W. Hong and Y.W Cho's web application project.<br></br>
                    We try to show you various computer vision applications: image classification, <br></br>
                    object detection, and instant segmentation.
                    <br></br>
                    <br></br>
                    You can navigate each of categories on the menu button on the left side <br></br>
                    <Icon name="hand point left" /><Icon name="hand point left" /><Icon name="hand point left" /><Icon name="hand point left" />
                    <br></br>
                    You can email us for your feedback and questions.
                    Thank you <Icon name="user secret" /> <br></br><br></br>
                    <Icon name="address book" /> kh4vv@virginia.edu (K.W Hong) <br></br>
                    <Icon name="address book outline" /> choyoungwoon@gmail.com (Y. W Cho)

                </p>
            </div>
        );
    }


}

export default Welcome;