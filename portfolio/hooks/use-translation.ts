export function useTranslation<T>(translations: T | undefined, language: "es" | "en") {
  return translations ?? {}
}

