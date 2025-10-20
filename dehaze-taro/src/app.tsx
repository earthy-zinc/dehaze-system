import React from 'react'
import { Provider } from 'react-redux'

import './app.less'
import configRequest from '@/utils/request'
import configStore from '@/store'

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
