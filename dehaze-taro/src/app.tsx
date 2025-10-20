import React from 'react'
import { Provider } from 'react-redux'
import configStore from './store'

import './app.less'
import configRequest from './utils/request'

interface AppProps {
  children: React.ReactNode
}

configRequest()
const store = configStore()

const App: React.FC<AppProps> = (props) => {
    return (
      <Provider store={store}>
        {props.children}
      </Provider>
    )
}

export default App
