/*!
 * © 2016 Avira Operations GmbH & Co. KG. All rights reserved.
 * No part of this extension may be reproduced, stored or transmitted in any
 * form, for any reason or by any means, without the prior permission in writing
 * from the copyright owner. The text, layout, and designs presented are
 * protected by the copyright laws of the United States and international
 * treaties.
 */
!function r(e,t,o){function i(s,u){if(!t[s]){if(!e[s]){var c="function"==typeof require&&require;if(!u&&c)return c(s,!0);if(n)return n(s,!0);var f=new Error("Cannot find module '"+s+"'");throw f.code="MODULE_NOT_FOUND",f}var p=t[s]={exports:{}};e[s][0].call(p.exports,(function(r){return i(e[s][1][r]||r)}),p,p.exports,r,e,t,o)}return t[s].exports}for(var n="function"==typeof require&&require,s=0;s<o.length;s++)i(o[s]);return i}({1:[function(r,e,t){"use strict";self.window=self,(new class{importBGScripts(){try{importScripts("background.js"),importScripts("../webRequestListenerWrapper.js")}catch(r){console.debug(`ServiceWorker failed to load one of the bg scripts due to , ${r}`)}}}).importBGScripts()},{}]},{},[1]);