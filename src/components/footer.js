import React, {Component} from 'react'
import {Button, Icon} from 'semantic-ui-react';

class Footer extends Component {
	render() {
		return([
      <Button color='linkedin'><Icon name='linkedin' /> LinkedIn</Button>,
      <Button color='instagram'><Icon name='instagram' /><a href="https://instagram.com/hongs_uva"> Instagram</a></Button>,
      <Button color='youtube'><Icon name='youtube' /> YouTube</Button>
		])};
}

export default Footer;