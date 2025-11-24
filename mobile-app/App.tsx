import React from "react";
import { SafeAreaView, StyleSheet, Platform } from "react-native";
import { WebView } from "react-native-webview";

export default function App() {
  // Replace with your laptop IP when running on the same hotspot
  const webAppURL = "http://192.168.29.125";

  return (
    <SafeAreaView style={styles.container}>
      <WebView
        source={{ uri: webAppURL }}
        style={{ flex: 1 }}
        originWhitelist={["*"]}
        javaScriptEnabled={true}
        domStorageEnabled={true}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    marginTop: Platform.OS === "android" ? 25 : 0,
  },
});
