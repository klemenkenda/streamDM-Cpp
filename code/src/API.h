#ifndef API_H_
#define API_H_

#ifdef STREAMDM_EXPORTS
#define STREAMDM_API __declspec(dllexport)
#else
#define STREAMDM_API __declspec(dllimport)
#endif

#endif /* API_H_ */
