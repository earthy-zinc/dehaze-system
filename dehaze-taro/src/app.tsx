import configStore from '@/store'
import configRequest from '@/utils/request'
import React from 'react'
import {Provider} from 'react-redux'

import './app.less'

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
