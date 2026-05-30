import { defineCollection, z } from 'astro:content';

const papers = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    originalTitle: z.string(),
    originalUrl: z.string().url(),
    authors: z.string(),
    institution: z.string().optional(),
    hfVotes: z.number().optional(),
    publishDate: z.string(),
    reviewDate: z.string(),
    tags: z.array(z.string()).default([]),
    description: z.string().optional(),
  }),
});

export const collections = { papers };
