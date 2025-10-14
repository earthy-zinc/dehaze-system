import React from 'react'
import { Provider } from 'react-redux'

import configStore from './store'

import './app.less'

const store = configStore()

interface AppProps {
  children: React.ReactNode
}

const App: React.FC<AppProps> = (props) => {
    return (
      <Provider store={store}>
        {props.children}
      </Provider>
    )
}

export default App
